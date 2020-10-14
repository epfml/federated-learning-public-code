# -*- coding: utf-8 -*-
import os

from PIL import Image
import torch.utils.data as data


class ImageNetDS(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset.
        img_size (int): Dimensions of the images: 128.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # get the filenames.
        self.class_paths = [
            (_class, os.path.join(self.root, _class))
            for _class in os.listdir(self.root)
        ]
        self.filenames = []
        self.filename2target = {}
        for _class, class_path in self.class_paths:
            for file_path in os.listdir(class_path):
                abs_file_path = os.path.join(class_path, file_path)
                self.filenames.append(abs_file_path)
                self.filename2target[abs_file_path] = _class

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.filenames[index])
        target = self.filename2target[self.filenames[index]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.filenames)
