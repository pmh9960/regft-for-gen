import os
import pickle
import random
from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames


class FastImageFolderWithPaths(ImageFolderWithPaths):
    # for fast debugging
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        cache_file = os.path.join(os.path.dirname(directory), os.path.basename(directory) + '.pkl')
        # load cache file if exists
        if os.path.isfile(cache_file):
            print(f'cache file found at {cache_file}')
            with open(cache_file, 'rb') as f:
                samples = pickle.load(f)
            return samples

        print(f'cache file not found, creating new one at {cache_file}')
        samples = ImageFolderWithPaths.make_dataset(directory, class_to_idx, extensions, is_valid_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)
        return samples


class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = FastImageFolderWithPaths(
            traindir,
            transform=self.preprocess)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return FastImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'


class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass


class ImageNetShuffledClasses(ImageNet):
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=32, num_workers=32, classnames='openai'):
        shuffled_classes_path = os.path.join(location, self.name(), 'imagenet_shuffled_classes.txt')
        if not os.path.isfile(shuffled_classes_path):
            classes = list(range(1000))
            random.shuffle(classes)
            with open(shuffled_classes_path, 'w') as f:
                f.write('\n'.join([str(x) for x in classes]))
        with open(shuffled_classes_path, 'r') as f:
            shuffled_classes = f.readlines()
        shuffled_classes = [int(x.rstrip('\n')) for x in shuffled_classes]
        self.target_transform = lambda x, shuffled_classes=shuffled_classes: shuffled_classes[x]
        super().__init__(preprocess, location, batch_size, num_workers, classnames)

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        target_transform = None if self.preprocess is None else self.target_transform  # ! hardcoding
        self.train_dataset = FastImageFolderWithPaths(traindir, transform=self.preprocess,
                                                      target_transform=target_transform)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_dataset(self):
        target_transform = None if self.preprocess is None else self.target_transform  # ! hardcoding
        return FastImageFolderWithPaths(self.get_test_path(), transform=self.preprocess,
                                        target_transform=target_transform)


class ImageNetK(ImageNet):

    def get_train_sampler(self):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler


class ImageNetSmallVal(ImageNet):

    def get_test_dataset(self):
        dataset = FastImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)
        dataset.samples = self.generate_fewshot_dataset(dataset.samples, num_shots=5, shuffle=False)
        return dataset

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True, shuffle=True,
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    if shuffle:
                        sampled_items = random.sample(items, num_shots)
                    else:
                        sampled_items = sorted(items)[:num_shots]
                else:
                    raise ValueError('Not enough samples', label, len(items))
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """for split samples (path, target)"""
        output = defaultdict(list)

        for item in data_source:
            output[item[1]].append(item)

        return output


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)


class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


ks = [1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 600]

for k in ks:
    cls_name = f"ImageNet{k}"
    dyn_cls = type(cls_name, (ImageNetK, ), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls

    cls_name = f"ImageNetShuffledClasses{k}"
    dyn_cls = type(cls_name, (ImageNetShuffledClasses, ImageNetK), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls
