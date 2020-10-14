# -*- coding: utf-8 -*-
import argparse
import cv2
import pickle
import os

import numpy as np
from tensorpack.dataflow import PrefetchDataZMQ, LMDBSerializer


def get_args():
    parser = argparse.ArgumentParser(description="aug data.")

    # define arguments.
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--data_type", default="train", type=str)
    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--force_delete", default=0, type=int)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    return args


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo)
    return dict


def sequential_downsampled_imagenet(args):
    data = DownsampledImageNet(args.data_dir, args.data_type, args.img_size)
    lmdb_file_path = os.path.join(
        args.data_dir, f"imagenet{args.img_size}_{args.data_type}.lmdb"
    )

    # delete file if exists.
    if os.path.exists(lmdb_file_path) and args.force_delete == 1:
        os.remove(lmdb_file_path)

    # serialize to the target path.
    ds1 = PrefetchDataZMQ(data, num_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


class DownsampledImageNet(object):
    def __init__(self, root_path, data_type, img_size=32):
        self.img_size = img_size
        self.img_size_square = self.img_size * self.img_size
        folder_path = os.path.join(root_path, f"imagenet{img_size}")

        # get dataset.
        list_of_data = [
            unpickle(os.path.join(folder_path, file))
            for file in os.listdir(folder_path)
            if ("train" if "train" in data_type else "val") in file
        ]
        mean_of_image = unpickle(
            [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if "train" in file
            ][0]
        )["mean"]

        # extract features.
        self.features, self.labels = self._get_images_and_labels(
            list_of_data, mean_of_image=mean_of_image
        )

    def _get_images_and_labels(self, list_of_data, mean_of_image):
        def _helper(_feature, _target, _mean):
            # process data.
            # _feature = _feature - _mean
            _target = [x - 1 for x in _target]
            return _feature, _target

        features, labels = [], []
        for _data in list_of_data:
            # extract raw data.
            _feature = _data["data"]
            _target = _data["labels"]
            _mean = mean_of_image

            # get data.
            feature, target = _helper(_feature, _target, _mean)

            # store data.
            features.append(feature)
            labels.append(target)

        features = np.concatenate(features)
        labels = np.concatenate(labels)
        return features, labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            if self.features[k] is not None and self.labels[k] is not None:
                # feature = cv2.imencode(".jpeg", self.features[k])[1]
                yield [self.features[k], self.labels[k]]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        pass


def main(args):
    sequential_downsampled_imagenet(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
