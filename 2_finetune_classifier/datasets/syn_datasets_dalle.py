
import os
import pickle
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .oxford_pets import OxfordPets
from .utils import DatasetBase, Datum, build_data_loader, read_json, write_json


class Dalle_Caltech(DatasetBase):

    dataset_dir = 'dalle_caltech_101'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_caltech.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_Cars(DatasetBase):

    dataset_dir = 'dalle_stanford_cars'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'cars_train')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_cars.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_DTD(DatasetBase):

    dataset_dir = 'dalle_dtd'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_dtd.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_Eurosat(DatasetBase):

    dataset_dir = 'dalle_eurosat'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_eurosat.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_fgvc(DatasetBase):

    dataset_dir = 'dalle_fgvc_aircraft'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_fgvc.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_Flowers(DatasetBase):

    dataset_dir = 'dalle_oxford_flowers'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_flower.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_Food(DatasetBase):

    dataset_dir = 'dalle_food-101'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_food.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_Pets(DatasetBase):

    dataset_dir = 'dalle_oxford_pets'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_pet.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_Sun(DatasetBase):

    dataset_dir = 'dalle_sun397'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_sun.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Dalle_UCF(DatasetBase):

    dataset_dir = 'dalle_ucf101'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'ucf101_midframes')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_ucf.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class DatasetWrapper(Dataset):
    def __init__(self, data_source, preprocess):
        self.data_source = data_source
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        output = {'labels': item.label, 'image_paths': item.impath}
        img = Image.open(item.impath).convert('RGB')
        img = self.preprocess(img)
        output['images'] = img
        return output


class SynDataset:
    def name(self):
        return 'syn_datasets'

    num_shots = -1
    num_shots_val = -1
    has_val = False
    target_class = Dalle_Caltech

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4,
                 *args, **kwargs):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_cafo = self.target_class(os.path.dirname(os.path.join(self.location, self.name())), self.num_shots)
        self.classnames = self.dataset_cafo.classnames

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        sampler = self.get_train_sampler()
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_dataset = DatasetWrapper(self.dataset_cafo.train_x, preprocess=self.preprocess)
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        if self.has_val:
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
        dataset = ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess, target_transform=self.classname2idx)
        dataset.samples = self.generate_fewshot_dataset(dataset.samples, num_shots=self.num_shots_val, shuffle=False)
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


def name_factory(root: str, name: str):
    return lambda self: os.path.join(root, name)


root = "syn_datasets"
dataset_cfgs = [
    (Dalle_Caltech, 'dalle/cafo/dalle_caltech-101', 16, False),
    (Dalle_DTD, 'dalle/cafo/dalle_dtd', 16, False),
    (Dalle_Eurosat, 'dalle/cafo/dalle_eurosat', 16, False),
    (Dalle_fgvc, 'dalle/cafo/dalle_fgvc_aircraft', 16, False),
    (Dalle_Food, 'dalle/cafo/dalle_food-101', 16, False),
    (Dalle_Flowers, 'dalle/cafo/dalle_oxford_flowers', 16, False),
    (Dalle_Pets, 'dalle/cafo/dalle_oxford_pets', 16, False),
    (Dalle_Cars, 'dalle/cafo/dalle_stanford_cars', 16, False),
    (Dalle_Sun, 'dalle/cafo/dalle_sun397', 16, False),
    (Dalle_UCF, 'dalle/cafo/dalle_ucf101', 16, False),
]
for target_class, synset_name, num_shots, has_val in dataset_cfgs:
    num_shots_str = "full" if num_shots == -1 else f"{num_shots}shots"
    cls_name = f"SynDataset_{synset_name.replace('.', '_').replace('/', '__').replace('-', '_')}_{num_shots_str}"
    dyn_cls = type(
        cls_name,
        (SynDataset, ),
        {
            "name": name_factory(root, synset_name),
            "num_shots": num_shots,
            "num_shots_val": 5 if has_val else -1,
            "has_val": has_val,
            "target_class": target_class,
        },
    )
    globals()[cls_name] = dyn_cls
