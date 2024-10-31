import os
import pickle
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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


class ZeroshotDataset:
    # def name(self):
    #     return 'imagenet'
    dataset_class = Caltech101
    num_shots = -1
    num_shots_val = -1

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

        self.dataset_cafo = self.dataset_class(self.location, num_shots=self.num_shots)
        self.classnames = self.dataset_cafo.classnames

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        self.train_dataset = DatasetWrapper(self.dataset_cafo.train_x, preprocess=self.preprocess)
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
        self.test_dataset = DatasetWrapper(self.dataset_cafo.test, preprocess=self.preprocess)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    # def get_test_path(self):
    #     test_path = os.path.join(self.location, self.name(), 'val_in_folder')
    #     if not os.path.exists(test_path):
    #         test_path = os.path.join(self.location, self.name(), 'val')
    #     return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    # def get_test_dataset(self):
    #     return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)


root = "syn_datasets"
dataset_cfgs = [
    ('caltech_101', Caltech101),
    ('dtd', DescribableTextures),
    ('eurosat', EuroSAT),
    ('fgvc_aircraft', FGVCAircraft),
    ('food101', Food101),
    ('oxford_flowers', OxfordFlowers),
    ('oxford_pets', OxfordPets),
    ('stanford_cars', StanfordCars),
    ('sun397', SUN397),
    ('ucf101', UCF101),
]
for cls_name, dataset_class in dataset_cfgs:
    dyn_cls = type(
        cls_name,
        (ZeroshotDataset, ),
        {
            "dataset_class": dataset_class,
        },
    )
    globals()[cls_name] = dyn_cls

for cls_name, dataset_class in dataset_cfgs:
    dyn_cls = type(
        cls_name + '_64shots',
        (ZeroshotDataset, ),
        {
            "dataset_class": dataset_class,
            "num_shots": 64,
        },
    )
    globals()[cls_name + '_64shots'] = dyn_cls
