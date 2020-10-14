# -*- coding: utf-8 -*-
import os

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


import pcode.datasets.loader.imagenet_folder as imagenet_folder
import pcode.datasets.loader.pseudo_imagenet_folder as pseudo_imagenet_folder
from pcode.datasets.loader.svhn_folder import define_svhn_folder
from pcode.datasets.loader.femnist import define_femnist_folder
import pcode.utils.op_paths as op_paths

"""the entry for classification tasks."""


def _get_cifar(conf, name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            if not conf.use_fake_centering
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            if not conf.use_fake_centering
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    normalize = normalize if conf.pn_normalize else None

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_cinic(conf, name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # download dataset.
    if download:
        # create the dir.
        op_paths.build_dir(root, force=False)

        # check_integrity.
        is_valid_download = True
        for _type in ["train", "valid", "test"]:
            _path = os.path.join(root, _type)
            if len(os.listdir(_path)) == 10:
                num_files_per_folder = [
                    len(os.listdir(os.path.join(_path, _x))) for _x in os.listdir(_path)
                ]
                num_files_per_folder = [x == 9000 for x in num_files_per_folder]
                is_valid_download = is_valid_download and all(num_files_per_folder)
            else:
                is_valid_download = False

        if not is_valid_download:
            # download.
            torchvision.datasets.utils.download_and_extract_archive(
                url="https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz",
                download_root=root,
                filename="cinic-10.tar.gz",
                md5=None,
            )
        else:
            print("Files already downloaded and verified.")

    # decide normalize parameter.
    normalize = transforms.Normalize(
        mean=(0.47889522, 0.47227842, 0.43047404),
        std=(0.24205776, 0.23828046, 0.25874835),
    )
    normalize = normalize if conf.pn_normalize else None

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return torchvision.datasets.ImageFolder(root=root, transform=transform)


def _get_mnist(conf, root, split, transform, target_transform, download):
    is_train = split == "train"
    normalize = (
        transforms.Normalize((0.1307,), (0.3081,)) if conf.pn_normalize else None
    )

    transform = transforms.Compose(
        [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
    )
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_stl10(conf, name, root, split, transform, target_transform, download):
    # right now this function is only used for unlabeled dataset.
    is_train = split == "train"

    # try to extract the downsample size if it has
    downsampled_size = conf.img_resolution

    # define the normalization operation.
    normalize = (
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if conf.pn_normalize
        else None
    )

    if is_train:
        split = "train+unlabeled"
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop((96, 96), 4)]
            + (
                [torchvision.transforms.Resize((downsampled_size, downsampled_size))]
                if downsampled_size is not None
                else []
            )
            + [transforms.ToTensor()]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            (
                [torchvision.transforms.Resize((downsampled_size, downsampled_size))]
                if downsampled_size is not None
                else []
            )
            + [transforms.ToTensor()]
            + ([normalize] if normalize is not None else [])
        )
    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_svhn(conf, root, split, transform, target_transform, download):
    is_train = split == "train"
    normalize = (
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if conf.pn_normalize
        else None
    )

    transform = transforms.Compose(
        [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
    )
    return define_svhn_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_femnist(conf, root, split, transform, target_transform, download):
    is_train = split == "train"
    assert (
        conf.pn_normalize is False
    ), "we've already normalize the image betwewen 0 and 1"

    transform = transforms.Compose([transforms.ToTensor()])
    return define_femnist_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_imagenet(conf, name, datasets_path, split):
    is_train = split == "train"
    is_downsampled = "8" in name or "16" in name or "32" in name or "64" in name
    root = os.path.join(
        datasets_path, "lmdb" if not is_downsampled else "downsampled_lmdb"
    )

    # get transform for is_downsampled=True.
    if is_downsampled:
        normalize = (
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            if conf.pn_normalize
            else None
        )

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                + ([normalize] if normalize is not None else [])
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
            )
    else:
        transform = None

    if conf.use_lmdb_data:
        if is_train:
            root = os.path.join(
                root, "{}train.lmdb".format(name + "_" if is_downsampled else "")
            )
        else:
            root = os.path.join(
                root, "{}val.lmdb".format(name + "_" if is_downsampled else "")
            )
        return imagenet_folder.define_imagenet_folder(
            conf=conf,
            name=name,
            root=root,
            flag=True,
            cuda=conf.graph.on_cuda,
            transform=transform,
            is_image=True and not is_downsampled,
        )
    else:
        return imagenet_folder.ImageNetDS(
            root=root, img_size=int(name[8:]), train=is_train, transform=transform
        )


def _get_pseudo_imagenet(conf, root, split="train"):
    is_train = split == "train"
    assert is_train

    # define normalize.
    normalize = (
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if conf.pn_normalize  # map to [-1, 1].
        else None
    )
    # define the transform.
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((112, 112), 4)]
        + (
            [transforms.Resize((conf.img_resolution, conf.img_resolution))]
            if conf.img_resolution is not None
            else []
        )
        + [transforms.ToTensor()]
        + ([normalize] if normalize is not None else [])
    )
    # return the dataset.
    return pseudo_imagenet_folder.ImageNetDS(
        root=root, train=is_train, transform=transform
    )


"""the entry for different supported dataset."""


def get_dataset(
    conf,
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == "cifar10" or name == "cifar100":
        return _get_cifar(
            conf, name, root, split, transform, target_transform, download
        )
    elif name == "cinic":
        return _get_cinic(
            conf, name, root, split, transform, target_transform, download
        )
    elif "stl10" in name:
        return _get_stl10(
            conf, name, root, split, transform, target_transform, download
        )
    elif name == "svhn":
        return _get_svhn(conf, root, split, transform, target_transform, download)
    elif name == "mnist":
        return _get_mnist(conf, root, split, transform, target_transform, download)
    elif name == "femnist":
        return _get_femnist(conf, root, split, transform, target_transform, download)
    elif "pseudo_imagenet" in name:
        return _get_pseudo_imagenet(conf, root, split)
    elif "imagenet" in name:
        return _get_imagenet(conf, name, datasets_path, split)
    else:
        raise NotImplementedError
