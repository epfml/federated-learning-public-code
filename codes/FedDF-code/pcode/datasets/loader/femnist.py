# -*- coding: utf-8 -*-
import os
import math
import json
import tarfile
import pickle
import hashlib
import random
import warnings
import collections

import numpy as np
from PIL import Image

import torch.utils.data as data
import torchvision
import torchvision.datasets.utils as data_utils


def define_femnist_folder(root, is_train, transform, target_transform, download):
    return FEMNIST(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        is_download=download,
    )


class FEMNIST(data.Dataset):
    """`FEMNIST <https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/>` Dataset.

    The code is heavily borrowed from
        https://github.com/TalwalkarLab/leaf/blob/master/data/,
    but with slightly adapted logic.
    """

    download_urls = {
        "by_class.zip": "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
        "by_write.zip": "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip",
    }

    def __init__(
        self,
        root,
        is_train,
        transform,
        target_transform,
        is_download,
        is_iid_sample=False,
        user_fraction=0.01,
        data_fraction=0.5,
        min_samples_per_user=128,
        split_by_sample=True,
        train_split_ratio=0.9,
    ):
        self.root = root
        self.train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.download = is_download

        self.data_fraction = data_fraction
        self.is_iid_sample = is_iid_sample
        self.min_samples_per_user = min_samples_per_user
        # the fraction of users in iid dataset; only valid if is_iid_sample is True.
        self.user_fraction = user_fraction
        self.split_by_sample = split_by_sample
        self.train_split_ratio = train_split_ratio
        self.img_size = 28

        # download/load the data.
        assert (
            is_iid_sample is False and split_by_sample is True
        ), "this data loader currently can only deal with niid case (and dataset is splitted by sample)."
        self.rng_state = random.Random(7)
        self.download_and_process_data()

    def _process_data(self, final_data_path):
        # get file dirs.
        # create the intermediate dir.
        intermediate_path = os.path.join(self.root, "intermediate")
        if not os.path.exists(intermediate_path):
            data_utils.makedir_exist_ok(intermediate_path)

        # extract file directories of images (by class).
        class_file_dirs_path = os.path.join(intermediate_path, "class_file_dirs.pickle")
        if not os.path.exists(class_file_dirs_path):
            print("extract file directories of images by class.")
            class_files = []  # (class, file directory)

            # init dir.
            class_dir = os.path.join(self.root, "by_class")
            classes = [c for c in os.listdir(class_dir) if len(c) == 2]

            # extract files.
            for cl in classes:
                cldir = os.path.join(class_dir, cl)
                subcls = [
                    s for s in os.listdir(cldir) if (("hsf" in s) and ("mit" not in s))
                ]

                for subcl in subcls:
                    subcldir = os.path.join(cldir, subcl)
                    image_dirs = [
                        os.path.join(subcldir, i) for i in os.listdir(subcldir)
                    ]

                    for image_dir in image_dirs:
                        class_files.append((cl, image_dir))
            print(
                f"extract file by class: # of samples={len(class_files)}. saving to path={class_file_dirs_path}."
            )
            save_obj(class_files, class_file_dirs_path)

        # extract file directories of images (by user).
        writer_file_dirs_path = os.path.join(
            intermediate_path, "writer_file_dirs.pickle"
        )
        if not os.path.exists(writer_file_dirs_path):
            print("extract file directories of images by writer.")
            writer_files = []  # (writer, file directory)
            writer_dir = os.path.join(self.root, "by_write")
            writer_parts = os.listdir(writer_dir)

            # init dir.
            for writer_part in writer_parts:
                writers_dir = os.path.join(writer_dir, writer_part)
                writers = os.listdir(writers_dir)

                for writer in writers:
                    _writer_dir = os.path.join(writers_dir, writer)
                    wtypes = os.listdir(_writer_dir)

                    for wtype in wtypes:
                        type_dir = os.path.join(_writer_dir, wtype)
                        images = os.listdir(type_dir)
                        image_dirs = [os.path.join(type_dir, i) for i in images]

                        for image_dir in image_dirs:
                            writer_files.append((writer, image_dir))
            print(
                f"extract file by writer: # of samples={len(writer_files)}. saving to path={writer_file_dirs_path}."
            )
            save_obj(writer_files, writer_file_dirs_path)

        # get the hash.
        # get the hash for class.
        class_file_hashes_path = os.path.join(
            intermediate_path, "class_file_hashes.pickle"
        )
        if not os.path.exists(class_file_hashes_path):
            # init.
            count = 0
            class_file_hashes = []
            class_file_dirs = load_obj(class_file_dirs_path)
            print("get the image hashes (by class).")

            # get the hashes.
            for tup in class_file_dirs:
                if count % 100000 == 0:
                    print("\thashed %d class images" % count)

                (cclass, cfile) = tup
                chash = hashlib.md5(open(cfile, "rb").read()).hexdigest()
                class_file_hashes.append((cclass, cfile, chash))
                count += 1
            save_obj(class_file_hashes, class_file_hashes_path)

        # get the hash for writer.
        writer_file_hashes_path = os.path.join(
            intermediate_path, "writer_file_hashes.pickle"
        )
        if not os.path.exists(writer_file_hashes_path):
            # init.
            count = 0
            writer_file_hashes = []
            writer_file_dirs = load_obj(writer_file_dirs_path)
            print("get the image hashes (by writer).")

            for tup in writer_file_dirs:
                if count % 100000 == 0:
                    print("hashed %d write images" % count)

                (cclass, cfile) = tup
                chash = hashlib.md5(open(cfile, "rb").read()).hexdigest()
                writer_file_hashes.append((cclass, cfile, chash))
                count += 1
            save_obj(writer_file_hashes, writer_file_hashes_path)

        # check the hash and assign class labels to writer.
        class_file_hashes = load_obj(
            class_file_hashes_path
        )  # each elem is (class, file dir, hash)
        writer_file_hashes = load_obj(
            writer_file_hashes_path
        )  # each elem is (writer, file dir, hash)
        writer_with_class_path = os.path.join(
            intermediate_path, "writer_with_class.pickle"
        )
        if not os.path.exists(writer_with_class_path):
            print("assigning class labels to writer images.")
            class_hash_dict = {}
            for i in range(len(class_file_hashes)):
                (c, f, h) = class_file_hashes[len(class_file_hashes) - i - 1]
                class_hash_dict[h] = (c, f)
            writer_classes = []
            for tup in writer_file_hashes:
                (w, f, h) = tup
                writer_classes.append((w, f, class_hash_dict[h][0]))
            save_obj(writer_classes, writer_with_class_path)

        # grouping images by writer.
        writer_class = load_obj(writer_with_class_path)
        images_by_writer_path = os.path.join(
            intermediate_path, "images_by_writer.pickle"
        )
        if not os.path.exists(images_by_writer_path):
            print("write images_by_writer")
            # each entry is a (writer, [list of (file, class)]) tuple
            writers = []
            cimages = []
            (cw, _, _) = writer_class[0]
            for (w, f, c) in writer_class:
                if w != cw:
                    writers.append((cw, cimages))
                    cw = w
                    cimages = [(f, c)]
                cimages.append((f, c))
            writers.append((cw, cimages))
            # save obj.
            save_obj(writers, images_by_writer_path)

        # create for the final data json.
        # converts a list of (writer, [list of (file,class)]) tuples into a json object
        # of the form:
        #   {users: [bob, etc], num_samples: [124, etc.],
        #   user_data: {bob : {x:[img1,img2,etc], y:[class1,class2,etc]}, etc}}
        # where 'img_' is a vectorized representation of the corresponding image.
        def relabel_class(c):
            """
            maps hexadecimal class value (string) to a decimal number
            returns:
            - 0 through 9 for classes representing respective numbers
            - 10 through 35 for classes representing respective uppercase letters
            - 36 through 61 for classes representing respective lowercase letters
            """
            if c.isdigit() and int(c) < 40:
                return int(c) - 30
            elif int(c, 16) <= 90:  # uppercase
                return int(c, 16) - 55
            else:
                return int(c, 16) - 61

        def write_to_json_file(users, num_samples, user_data, json_index):
            all_data = {}
            all_data["users"] = users
            all_data["num_samples"] = num_samples
            all_data["user_data"] = user_data

            file_name = "all_data_%d.json" % json_index
            file_path = os.path.join(final_data_path, file_name)
            print("writing %s" % file_name)
            jump_json(all_data, file_path)

        def write_to_json_files(all_writers, max_writers):
            writer_count = 0
            json_index = 0
            users = []
            num_samples = []
            user_data = {}

            for (w, l) in all_writers:
                users.append(w)
                num_samples.append(len(l))
                user_data[w] = {"x": [], "y": []}

                size = self.img_size, self.img_size  # original image size is 128, 128
                for (f, c) in l:
                    img = Image.open(f)
                    gray = img.convert("L")
                    gray.thumbnail(size, Image.ANTIALIAS)
                    arr = np.asarray(gray).copy()
                    vec = arr.flatten()
                    vec = vec / 255  # scale all pixel values to between 0 and 1
                    vec = vec.tolist()
                    nc = relabel_class(c)

                    user_data[w]["x"].append(vec)
                    user_data[w]["y"].append(nc)
                writer_count += 1

                # redirect a new json file.
                if writer_count == max_writers:
                    write_to_json_file(users, num_samples, user_data, json_index)

                    # reinit.
                    writer_count = 0
                    json_index += 1
                    users[:] = []
                    num_samples[:] = []
                    user_data.clear()

            # in case we have something left.
            if writer_count > 0:
                write_to_json_file(users, num_samples, user_data, json_index)

        # start the final processing.
        if not os.path.exists(final_data_path):
            data_utils.makedir_exist_ok(final_data_path)

        MAX_WRITERS = 100  # max number of writers per json file.
        writers = load_obj(images_by_writer_path)
        num_json_files = int(math.ceil(len(writers) / MAX_WRITERS))

        if (
            len([x for x in os.listdir(final_data_path) if "json" in x])
            != num_json_files
        ):
            print(
                f"final step for creating all data 1: save the json files to disks. we have {num_json_files} json files."
            )
            write_to_json_files(writers, MAX_WRITERS)

        if not os.path.exists(self.all_data_tgz_file):
            print(f"final step for creating all data 2: save them to tgz file.")
            tar_compress_folder(final_data_path)

    def _sample_data(self):
        new_user_count = 0  # for iid case
        folder_name = os.path.join(
            self.root,
            f"sampled_{'iid' if self.is_iid_sample else 'niid'}_data_fraction-{self.data_fraction}{f'_user_fraction-{self.user_fraction}' if self.is_iid_sample else ''}",
        )
        folder_tgz = folder_name + ".tgz"

        # build or extract.
        print(f"sample data and will save to {folder_name}")
        data_utils.makedir_exist_ok(folder_name)

        is_finished_sampling = len(os.listdir(folder_name)) == len(
            os.listdir(self.all_data_path)
        )
        if os.path.exists(folder_tgz) and is_finished_sampling:
            print("has finished sampling and compressed the sampled data.")
            return folder_name
        elif os.path.exists(folder_tgz):
            print(
                "the sampling has not been finished (but we have the tgz file). So let's decompress the tgz file."
            )
            tar_decompress_folder(folder_tgz)
            return folder_name

        # (rough) check the number of files in folder_name.
        if not is_finished_sampling:
            print("the number of sampled json file is incorrect. sample it again.")

            for f in os.listdir(self.all_data_path):
                file_dir = os.path.join(self.all_data_path, f)
                with open(file_dir, "r") as inf:
                    # Load data into an OrderedDict, to prevent ordering changes
                    # and enable reproducibility
                    data = json.load(inf, object_pairs_hook=collections.OrderedDict)

                # get some meta info.
                num_users = len(data["users"])
                tot_num_samples = sum(data["num_samples"])
                num_new_samples = int(self.data_fraction * tot_num_samples)
                hierarchies = None

                # if it is iid sample.
                if self.is_iid_sample:
                    raw_list = list(data["user_data"].values())
                    raw_x = [elem["x"] for elem in raw_list]
                    raw_y = [elem["y"] for elem in raw_list]
                    x_list = [
                        item for sublist in raw_x for item in sublist
                    ]  # flatten raw_x
                    y_list = [
                        item for sublist in raw_y for item in sublist
                    ]  # flatten raw_y

                    # get new users and new indices.
                    num_new_users = max(int(round(self.user_fraction * num_users)), 1)
                    indices = [i for i in range(tot_num_samples)]
                    new_indices = self.rng_state.sample(indices, num_new_samples)
                    users = [str(i + new_user_count) for i in range(num_new_users)]

                    # get the new data and divide them (iid).
                    user_data = dict(
                        (user, collections.defaultdict(list)) for user in users
                    )
                    all_x_samples = [x_list[i] for i in new_indices]
                    all_y_samples = [y_list[i] for i in new_indices]
                    x_groups = iid_divide(all_x_samples, num_new_users)
                    y_groups = iid_divide(all_y_samples, num_new_users)

                    # assign the info.
                    for i in range(num_new_users):
                        user_data[users[i]]["x"] = x_groups[i]
                        user_data[users[i]]["y"] = y_groups[i]
                    num_samples = [len(user_data[u]["y"]) for u in users]
                    new_user_count += num_new_users
                else:
                    ctot_num_samples = 0

                    users = data["users"]
                    users_and_hiers = None
                    if "hierarchies" in data:
                        users_and_hiers = list(zip(users, data["hierarchies"]))
                        self.rng_state.shuffle(users_and_hiers)
                        hierarchies = []
                    else:
                        self.rng_state.shuffle(users)

                    # init for the sampling (by user).
                    user_i = 0
                    num_samples = []
                    user_data = {}

                    # sample by user.
                    while ctot_num_samples < num_new_samples:
                        if users_and_hiers is not None:
                            user, hier = users_and_hiers[user_i]
                            hierarchies.append(hier)
                        else:
                            user = users[user_i]

                        cdata = data["user_data"][user]
                        cnum_samples = len(data["user_data"][user]["y"])
                        if ctot_num_samples + cnum_samples > num_new_samples:
                            cnum_samples = num_new_samples - ctot_num_samples
                            indices = [i for i in range(cnum_samples)]
                            new_indices = self.rng_state.sample(indices, cnum_samples)
                            x = []
                            y = []
                            for i in new_indices:
                                x.append(data["user_data"][user]["x"][i])
                                y.append(data["user_data"][user]["y"][i])
                            cdata = {"x": x, "y": y}

                        num_samples.append(cnum_samples)
                        user_data[user] = cdata

                        ctot_num_samples += cnum_samples
                        user_i += 1

                    if "hierarchies" in data:
                        users = [u for u, h in users_and_hiers][:user_i]
                    else:
                        users = users[:user_i]

                # create the .json file.
                all_data = {}
                all_data["users"] = users
                if hierarchies is not None:
                    all_data["hierarchies"] = hierarchies
                all_data["num_samples"] = num_samples
                all_data["user_data"] = user_data

                # save to json file.
                file_path = os.path.join(folder_name, f)
                print(f"\tsave sampled json file to the path={file_path}.")
                jump_json(all_data, file_path=file_path)

        print(f"save data to tgz file.")
        tar_compress_folder(folder_name)
        return folder_name

    def _remove_invalid_user(self, data_path):
        if self.min_samples_per_user == 0:
            print("skip the filtering due to min_samples_per_user=0.")
            return data_path

        # build folder and filter user.
        folder_name = os.path.join(self.root, "filtered_" + data_path)
        folder_tgz = folder_name + ".tgz"

        # init.
        print(f"filter sampled data and will save to {folder_name}")
        data_utils.makedir_exist_ok(folder_name)

        is_finished_filtering = len(os.listdir(folder_name)) == len(
            os.listdir(self.all_data_path)
        )
        if os.path.exists(folder_tgz) and is_finished_filtering:
            print("has finished filtering and compressed the sampled data.")
            return folder_name

        # start filtering.
        if not is_finished_filtering:
            for f in [f for f in os.listdir(data_path) if f.endswith(".json")]:
                users = []
                hierarchies = []
                num_samples = []
                user_data = {}

                # load the data.
                file_dir = os.path.join(data_path, f)
                data = load_json(file_dir)

                num_users = len(data["users"])
                for i in range(num_users):
                    curr_user = data["users"][i]
                    curr_hierarchy = None
                    if "hierarchies" in data:
                        curr_hierarchy = data["hierarchies"][i]
                    curr_num_samples = data["num_samples"][i]
                    if curr_num_samples >= self.min_samples_per_user:
                        user_data[curr_user] = data["user_data"][curr_user]
                        users.append(curr_user)
                        if curr_hierarchy is not None:
                            hierarchies.append(curr_hierarchy)
                        num_samples.append(data["num_samples"][i])

                # save the valid data.
                all_data = {}
                all_data["users"] = users
                if len(hierarchies) == len(users):
                    all_data["hierarchies"] = hierarchies
                all_data["num_samples"] = num_samples
                all_data["user_data"] = user_data

                file_path = os.path.join(folder_name, f)
                print(f"\tsave filtered and sampled json file to the path={file_path}.")
                jump_json(all_data, file_path=file_path)

        print(f"save data to tgz file.")
        tar_compress_folder(folder_name)
        return folder_name

    def _split_data(self, data_path):
        print(
            f"split the data for the path={data_path}, split_by_sample={self.split_by_sample}."
        )

        if not self.split_by_sample:  # i.e. we will split by user.
            # 1 pass through all the json files to instantiate arr
            # containing all possible (user, .json file name) tuples

            user_files = []
            for f in os.listdir(data_path):
                file_dir = os.path.join(data_path, f)
                with open(file_dir, "r") as inf:
                    # Load data into an OrderedDict, to prevent ordering changes
                    # and enable reproducibility
                    data = json.load(inf, object_pairs_hook=collections.OrderedDict)

                include_hierarchy = "hierarchies" in data
                if include_hierarchy:
                    user_files.extend(
                        [
                            (u, h, ns, f)
                            for (u, h, ns) in zip(
                                data["users"], data["hierarchies"], data["num_samples"]
                            )
                        ]
                    )
                else:
                    user_files.extend(
                        [
                            (u, ns, f)
                            for (u, ns) in zip(data["users"], data["num_samples"])
                        ]
                    )

            # randomly sample from user_files to pick training set users
            num_users = len(user_files)
            num_train_users = int(self.train_split_ratio * num_users)
            indices = [i for i in range(num_users)]
            train_indices = self.rng_state.sample(indices, num_train_users)
            train_blist = [False for i in range(num_users)]
            for i in train_indices:
                train_blist[i] = True

            train_user_files = []
            test_user_files = []
            for i in range(num_users):
                if train_blist[i]:
                    train_user_files.append(user_files[i])
                else:
                    test_user_files.append(user_files[i])

            # todo....
            assert False, "TODO..."
        else:
            train_data_path = data_path + "_train"
            test_data_path = data_path + "_test"
            train_meta_data_path = train_data_path + "_meta.json"
            test_meta_data_path = test_data_path + "_meta.json"

            is_finished_splitting = (
                os.path.exists(train_data_path)
                and os.path.exists(test_data_path)
                and len(os.listdir(train_data_path)) == len(os.listdir(test_data_path))
                and len(os.listdir(train_data_path)) > 100
                and os.path.exists(train_meta_data_path)
                and os.path.exists(test_meta_data_path)
            )
            if is_finished_splitting:
                print(f"exist the splitted files (exit).")
                return train_data_path, test_data_path

            print("\tsplitting the dataset into train/test by sample.")
            data_utils.makedir_exist_ok(train_data_path)
            data_utils.makedir_exist_ok(test_data_path)
            meta_train = collections.defaultdict(list)
            meta_test = collections.defaultdict(list)

            for f in os.listdir(data_path):
                file_dir = os.path.join(data_path, f)
                with open(file_dir, "r") as inf:
                    # Load data into an OrderedDict, to prevent ordering changes
                    # and enable reproducibility
                    data = json.load(inf, object_pairs_hook=collections.OrderedDict)

                print(f'\twe have {len(data["users"])} users.')
                for i, u in enumerate(data["users"]):
                    curr_num_samples = len(data["user_data"][u]["y"])
                    if curr_num_samples >= 2:
                        # ensures number of train and test samples both >= 1
                        num_train_samples = max(
                            1, int(self.train_split_ratio * curr_num_samples)
                        )
                        if curr_num_samples == 2:
                            num_train_samples = 1

                        indices = [j for j in range(curr_num_samples)]
                        train_indices = self.rng_state.sample(
                            indices, num_train_samples
                        )
                        test_indices = [
                            i for i in range(curr_num_samples) if i not in train_indices
                        ]

                        # if we have a valid train/test split.
                        if len(train_indices) >= 1 and len(test_indices) >= 1:
                            user_data_train = {"x": [], "y": []}
                            user_data_test = {"x": [], "y": []}

                            train_blist = [False for j in range(curr_num_samples)]
                            test_blist = [False for j in range(curr_num_samples)]
                            for j in train_indices:
                                train_blist[j] = True
                            for j in test_indices:
                                test_blist[j] = True

                            for j in range(curr_num_samples):
                                if train_blist[j]:
                                    user_data_train["x"].append(
                                        data["user_data"][u]["x"][j]
                                    )
                                    user_data_train["y"].append(
                                        data["user_data"][u]["y"][j]
                                    )
                                elif test_blist[j]:
                                    user_data_test["x"].append(
                                        data["user_data"][u]["x"][j]
                                    )
                                    user_data_test["y"].append(
                                        data["user_data"][u]["y"][j]
                                    )

                            # save the data to disk.
                            all_data_train = {
                                "user_data": user_data_train,
                                "hierarchies": data["hierarchies"][i]
                                if "hierarchies" in data
                                else None,
                            }
                            all_data_test = {
                                "user_data": user_data_test,
                                "hierarchies": data["hierarchies"][i]
                                if "hierarchies" in data
                                else None,
                            }
                            meta_train["users"].append(u)
                            meta_test["users"].append(u)
                            meta_train["num_samples"].append(len(user_data_train["x"]))
                            meta_test["num_samples"].append(len(user_data_test["x"]))

                            # save to path.
                            jump_json(
                                all_data_train,
                                os.path.join(train_data_path, f"{u}.json"),
                            )
                            jump_json(
                                all_data_test, os.path.join(test_data_path, f"{u}.json")
                            )
                print(f"\tsplitted {f}. processed {len(meta_train['users'])} users.")
            # save the meta data to the disk.
            jump_json(meta_train, train_meta_data_path)
            jump_json(meta_test, test_meta_data_path)
            return train_data_path, test_data_path

    def _load_meta_data_and_display_stat(self, splitted_data_paths):
        print(f"load the json for {'train' if self.train else 'test'}.")
        data_path = splitted_data_paths[0] if self.train else splitted_data_paths[1]
        tr_meta_data_path = splitted_data_paths[0] + "_meta.json"
        te_meta_data_path = splitted_data_paths[1] + "_meta.json"
        tr_meta_data = load_json(tr_meta_data_path)
        te_meta_data = load_json(te_meta_data_path)
        num_samples = tr_meta_data["num_samples"]
        print(
            f"stat: we have {len(num_samples)} users in total. the total # of samples: {sum(num_samples)}. the samples per device: mean={sum(num_samples) / len(num_samples)}, std={np.std(num_samples):.3f}."
        )
        return (
            data_path,
            {
                "tr_meta_data_path": tr_meta_data_path,
                "te_meta_data_path": te_meta_data_path,
            },
            {"tr_meta_data": tr_meta_data, "te_meta_data": te_meta_data},
        )

    def download_and_process_data(self):
        # if we need to download the all_data.
        if self.download:
            # create the root dir.
            data_utils.makedir_exist_ok(self.root)
            self.all_data_path = os.path.join(self.root, "all_data")
            self.all_data_tgz_file = self.all_data_path + ".tgz"

            if not os.path.exists(self.all_data_tgz_file):
                warnings.warn(
                    "The compresssed file is missing. It will take a while (at least hours) to download, uncompress and process the data."
                )

                # download and uncompress data.
                print("download and extract archive.")
                for name, url in self.download_urls.items():
                    torchvision.datasets.utils.download_and_extract_archive(
                        url=url, download_root=self.root, filename=name, md5=None
                    )

                # process the data.
                self._process_data(self.all_data_path)
            else:
                print("Files already downloaded.")
                if not os.path.exists(self.all_data_path):
                    tar_decompress_folder(self.all_data_tgz_file)

        # perform sampling from the all_data and remove invalid information.
        data_path = self._sample_data()
        # data_path = self._remove_invalid_user(data_path)

        # split the dataset.
        splitted_data_paths = self._split_data(data_path)

        # display the stat of the train/test data.
        (
            self.data_path,
            self.meta_data_path,
            self.meta_data,
        ) = self._load_meta_data_and_display_stat(splitted_data_paths)

    def set_user(self, client_id):
        tr_users = self.meta_data["tr_meta_data"]["users"]
        tr_num_samples = self.meta_data["tr_meta_data"]["num_samples"]
        tr_users_and_num_samples = list(zip(tr_users, tr_num_samples))

        # filter out users based on the train dataset.
        tr_users_and_num_samples = [
            (user, num_samples)
            for user, num_samples in tr_users_and_num_samples
            if num_samples > self.min_samples_per_user
        ]

        # extract the correct user.
        user, num_samples = tr_users_and_num_samples[client_id]

        # load the data.
        data_path = os.path.join(self.data_path, f"{user}.json")
        data = load_json(data_path)

        # build the self.data and self.target.
        self.data = data["user_data"]["x"]
        self.targets = data["user_data"]["y"]

        # some stat.
        self.data_size = len(self.data)
        unique_elements, counts_elements = np.unique(self.targets, return_counts=True)
        print(f"label stat: {list(zip(unique_elements, counts_elements))}")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        assert hasattr(self, "data") and hasattr(self, "targets")
        img, target = self.data[index], int(self.targets[index])
        img = np.array(img).reshape(self.img_size, self.img_size)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.data_size if hasattr(self, "data_size") else 0

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def load_json(file_dir):
    with open(file_dir, "r") as inf:
        data = json.load(inf)
    return data


def jump_json(json_file, file_path):
    with open(file_path, "w") as outfile:
        json.dump(json_file, outfile)


def tar_compress_folder(folder_path):
    with tarfile.open(folder_path + ".tgz", "w:gz") as tar:
        for name in os.listdir(folder_path):
            tar.add(os.path.join(folder_path, name))


def tar_decompress_folder(file_path):
    with tarfile.open(file_path) as tar:
        subdir_and_files = [tarinfo for tarinfo in tar.getmembers()]
        tar.extractall(members=subdir_and_files)


def iid_divide(l, g):
    """
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i : group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i : bi + group_size * (i + 1)])
    return glist


if __name__ == "__main__":
    f = FEMNIST(
        root="./data/femnist",
        is_train=True,
        transform=None,
        target_transform=None,
        is_download=True,
        min_samples_per_user=120,
    )

    f.set_user(client_id=0)
    f[1]
