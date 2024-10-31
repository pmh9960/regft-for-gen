import os
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from .imagenet import FastImageFolderWithPaths, ImageNet


class ImageNetSD(ImageNet):
    num_shots = -1
    num_shots_val = -1

    def name(self):
        return 'syn_datasets/IN1k_SD2.1_CFG5.0'

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = FastImageFolderWithPaths(
            traindir,
            transform=self.preprocess)

        self.train_dataset.samples = self.generate_fewshot_dataset(self.train_dataset.samples, num_shots=self.num_shots)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

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

    def populate_test(self):
        if self.has_val:
            super().populate_test()

    def get_test_dataset(self):
        dataset = FastImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)
        dataset.samples = self.generate_fewshot_dataset(dataset.samples, num_shots=self.num_shots_val, shuffle=False)
        return dataset


class ImageNetSD_selected(ImageNetSD):
    selected_samples_path = None

    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=32, num_workers=32, classnames='openai'):
        self.selected_samples_path = os.path.join(location, self.name(), self.selected_samples_path)
        super().__init__(preprocess, location, batch_size, num_workers, classnames)

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = FastImageFolderWithPaths(traindir, transform=self.preprocess)

        samples = [(str(sample[0]), int(sample[1])) for sample in np.load(self.selected_samples_path)['samples']]

        # self.train_dataset.samples = self.generate_fewshot_dataset(self.train_dataset.samples, num_shots=self.num_shots)
        self.train_dataset.samples = self.generate_fewshot_dataset(samples, num_shots=self.num_shots)
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )


def name_factory(root: str, name: str):
    return lambda self: os.path.join(root, name)

# autopep8: off
root = "syn_datasets"
num_shots_candidate = [-1] + [2 ** i for i in range(11)]
dataset_cfgs = []
# dataset_cfgs.append(('IN1k_SD2.1_CFG5.0', -1, True))
# dataset_cfgs.extend([('IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 256])
# dataset_cfgs.extend([('IN1k_SD2.1_CFG2.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 64])
# dataset_cfgs.extend([('IN1k_SD2.1_CFG3.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 8])
# dataset_cfgs.extend([('IN1k_SD2.1_CFG7.5', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 8])
# dataset_cfgs.extend([('IN1k_SD2.1_CFG10', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 8])
dataset_cfgs.extend([('a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_photo_of_a/IN1k_SD2.1_CFG1.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_photo_of_a/IN1k_SD2.1_CFG10.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_sculpture_of_a/IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_rendering_of_a/IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_sketch_of_a/IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_painting_of_a/IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('a_bad_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True) for num_shots in num_shots_candidate if num_shots <= 256])
dataset_cfgs.extend([('dalle/imagenet', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 16])
dataset_cfgs.extend([('SDv1.5/imagenet/CFG5.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('SDXL/imagenet/CFG5.0', num_shots, False) for num_shots in num_shots_candidate if num_shots <= 64])
# autopep8: on
for synset_name, num_shots, has_val in dataset_cfgs:
    num_shots_str = "full" if num_shots == -1 else f"{num_shots}shots"
    cls_name = f"ImageNetSD_{synset_name.replace('.', '_').replace('/', '__')}_{num_shots_str}"
    dyn_cls = type(
        cls_name,
        (ImageNetSD, ),
        {
            "name": name_factory(root, synset_name),
            "num_shots": num_shots,
            "has_val": has_val,
            "num_shots_val": 5 if has_val else -1,
        },
    )
    globals()[cls_name] = dyn_cls

# autopep8: off
dataset_cfgs = []
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 0, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 1, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 2, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 3, False) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 0, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 1, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 2, True) for num_shots in num_shots_candidate if num_shots <= 64])
dataset_cfgs.extend([('more_samples/a_photo_of_a/IN1k_SD2.1_CFG5.0', num_shots, True, 3, True) for num_shots in num_shots_candidate if num_shots <= 64])
# autopep8: on
for synset_name, num_shots, has_val, conf_level, considered_corr in dataset_cfgs:
    num_shots_str = "full" if num_shots == -1 else f"{num_shots}shots"
    if not considered_corr:
        cls_name = f"ImageNetSD_selected_{conf_level}_of_4_confidence_{synset_name.replace('.', '_').replace('/', '__')}_{num_shots_str}"
        selected_samples_path = f'selected_samples_{conf_level}_of_4_by_clip_vitb16_with_gpt_confidence.npz'
    else:
        cls_name = f"ImageNetSD_selected_{conf_level}_of_4_confidence_considered_correctness_{synset_name.replace('.', '_').replace('/', '__')}_{num_shots_str}"
        selected_samples_path = f'selected_samples_{conf_level}_of_4_by_clip_vitb16_with_gpt_confidence_considered_correctness.npz'
    dyn_cls = type(
        cls_name,
        (ImageNetSD_selected, ),
        {
            "name": name_factory(root, synset_name),
            "num_shots": num_shots,
            "has_val": has_val,
            "num_shots_val": 5 if has_val else -1,
            'selected_samples_path': selected_samples_path,
        },
    )
    globals()[cls_name] = dyn_cls
