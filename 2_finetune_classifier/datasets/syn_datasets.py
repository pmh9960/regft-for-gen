import os
import pickle
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .caltech101 import Caltech101
from .common import ImageFolderWithPaths, SubsetSampler
from .dtd import DescribableTextures
from .eurosat import EuroSAT
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .oxford_pets import OxfordPets
from .stanford_cars import StanfordCars
from .sun397 import SUN397
from .ucf101 import UCF101


class SynDataset:
    def name(self):
        return 'syn_datasets'

    num_shots = -1
    num_shots_val = -1
    has_val = False
    target_class = Caltech101

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

        self.classnames = self.target_class(location, 64).classnames
        self.sorted_classnames = sorted(self.classnames)
        def redefine_idx(idx):
            return self.classnames.index(self.sorted_classnames[idx])
        self.classname2idx = redefine_idx

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(traindir, transform=self.preprocess, target_transform=self.classname2idx)

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
    (Caltech101, 'caltech-101/IN1k_SD2.1_CFG5.0', 64, False),
    (DescribableTextures, 'dtd/IN1k_SD2.1_CFG5.0', 64, False),
    (EuroSAT, 'eurosat/IN1k_SD2.1_CFG5.0', 64, False),
    (FGVCAircraft, 'fgvc_aircraft/IN1k_SD2.1_CFG5.0', 64, False),
    (Food101, 'food-101/IN1k_SD2.1_CFG5.0', 64, False),
    (OxfordFlowers, 'oxford_flowers/IN1k_SD2.1_CFG5.0', 64, False),
    (OxfordPets, 'oxford_pets/IN1k_SD2.1_CFG5.0', 64, False),
    (StanfordCars, 'stanford_cars/IN1k_SD2.1_CFG5.0', 64, False),
    (SUN397, 'sun397/IN1k_SD2.1_CFG5.0', 64, False),
    (UCF101, 'ucf101/IN1k_SD2.1_CFG5.0', 64, False),
    # SDXL
    (Caltech101, 'SDXL/caltech-101/CFG5.0', 64, False),
    (DescribableTextures, 'SDXL/dtd/CFG5.0', 64, False),
    (EuroSAT, 'SDXL/eurosat/CFG5.0', 64, False),
    (FGVCAircraft, 'SDXL/fgvc_aircraft/CFG5.0', 64, False),
    (Food101, 'SDXL/food-101/CFG5.0', 64, False),
    (OxfordFlowers, 'SDXL/oxford_flowers/CFG5.0', 64, False),
    (OxfordPets, 'SDXL/oxford_pets/CFG5.0', 64, False),
    (StanfordCars, 'SDXL/stanford_cars/CFG5.0', 64, False),
    (SUN397, 'SDXL/sun397/CFG5.0', 64, False),
    (UCF101, 'SDXL/ucf101/CFG5.0', 64, False),
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
