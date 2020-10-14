# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.datasets.utils import check_integrity

from pcode.datasets.loader.preprocess_toolkit import get_transform
from pcode.datasets.loader.utils import LMDBPT


def define_imagenet_folder(
    conf, name, root, flag, cuda=True, transform=None, is_image=True
):
    is_train = "train" in root
    # note that for the standard imagenet training,
    # we should correctly normalize the input.
    if transform is None:
        transform = get_transform(name, augment=is_train, color_process=False)

    if flag:
        print("load imagenet from lmdb: {}".format(root))
        return LMDBPT(root, transform=transform, is_image=is_image)
    else:
        print("load imagenet using pytorch's default dataloader.")
        return datasets.ImageFolder(
            root=root, transform=transform, target_transform=None
        )


class ImageNetDS(data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.
    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    base_folder = "imagenet{}"
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]

    test_list = [["val_data", ""]]

    def __init__(
        self, root, img_size, train=True, transform=None, target_transform=None
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # now load the picked numpy arrays
        if self.train:
            self.data = []
            self.targets = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, "rb") as fo:
                    entry = pickle.load(fo)
                    self.data.append(entry["data"])
                    self.targets += [label - 1 for label in entry["labels"]]
                    self.mean = entry["mean"]

            self.data = np.concatenate(self.data)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            with open(file, "rb") as fo:
                entry = pickle.load(fo)
                self.data = entry["data"]
                self.targets = [label - 1 for label in entry["labels"]]

        self.data = self.data.reshape((self.data.shape[0], 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
