# -*- coding: utf-8 -*-
import math
import functools

import numpy as np

import torch
import torch.distributed as dist


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        if self.replaced_targets is None:
            return self.data[data_idx]
        else:
            return (self.data[data_idx][0], self.replaced_targets[index])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        # evaluate the the difference between original labels and the simulated labels.
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def clean_replaced_targets(self):
        self.replaced_targets = None


class DataSampler(object):
    def __init__(
        self, conf, data, data_scheme, data_percentage=None, selected_classes=None
    ):
        # init.
        self.conf = conf
        self.data = data
        self.data_size = len(self.data)
        self.data_scheme = data_scheme
        self.data_percentage = data_percentage
        self.selected_classes = selected_classes

        # get unshuffled indices.
        self.indices = np.array([x for x in range(0, self.data_size)])
        self.sampled_indices = None

    def sample_indices(self):
        if self.data_scheme == "random_sampling":
            self.sampled_indices = self.conf.random_state.choice(
                self.indices,
                size=int(self.data_size * self.data_percentage),
                replace=False,
            )
        elif self.data_scheme == "class_selection":
            self.sampled_indices = [
                index
                for index, target in enumerate(self.data.targets)
                if target in self.selected_classes
            ]
            if self.data_percentage is not None:
                self.sampled_indices = self.conf.random_state.choice(
                    self.sampled_indices,
                    size=int(len(self.sampled_indices) * self.data_percentage),
                    replace=False,
                )
        else:
            raise NotImplementedError(
                "this sampling scheme has not been supported yet."
            )

    def use_indices(self, sampled_indices=None):
        assert sampled_indices is not None or self.sampled_indices is not None
        return Partition(
            self.data,
            indices=sampled_indices
            if sampled_indices is not None
            else self.sampled_indices,
        )


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self, conf, data, partition_sizes, partition_type, consistent_indices=True
    ):
        # prepare info.
        self.conf = conf
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []

        # get data, data_size, indices of the data.
        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices

        # apply partition function.
        self.partition_indices(indices)

    def partition_indices(self, indices):
        if self.conf.graph.rank == 0:
            indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)

        # partition indices.
        from_index = 0
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index

        # display the class distribution over the partitions.
        if self.conf.graph.rank == 0:
            record_class_distribution(
                self.partitions, self.data.targets, print_fn=self.conf.logger.log
            )

    def _create_indices(self, indices):
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            # it will randomly shuffle the indices.
            self.conf.random_state.shuffle(indices)
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            num_classes = len(np.unique(self.data.targets))
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)

            list_of_indices = build_non_iid_by_dirichlet(
                random_state=self.conf.random_state,
                indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ]
                ),
                non_iid_alpha=self.conf.non_iid_alpha,
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )
        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            # sync the indices over clients.
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets, print_fn):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    print_fn(
        f"the histogram of the targets in the partitions: {targets_of_partitions.items()}"
    )
    return targets_of_partitions


num_classes = 10
num_indices = 50000
targets = np.array(
    functools.reduce(
        lambda a, b: a + b,
        [[idx] * int(num_indices / num_classes) for idx in range(num_classes)],
    )
)

# import time

# start_time = time.time()
# partitions = build_non_iid_by_dirichlet(
#     random_state=np.random.RandomState(7),
#     targets=targets,
#     non_iid_alpha=0.001,
#     num_classes=num_classes,
#     num_indices=num_indices,
#     n_workers=100,
#     n_auxi_workers=num_classes,
# )
# end_time = time.time()
# print(end_time - start_time)
# record_class_distribution(partitions, targets, print_fn=print)


def get_imagenet1k_classes(num_overlap_classes, random_state, num_total_classes=100):
    _selected_imagenet_classes = []
    _selected_cifar100_classes = []
    _cifar100_classes = list(cifar100_class_id_2_imagenet1k_class_id.keys())

    while len(_selected_imagenet_classes) < num_overlap_classes:
        random_state.shuffle(_cifar100_classes)

        _cifar100_class = _cifar100_classes[0]
        _imagenet_classes = cifar100_class_id_2_imagenet1k_class_id[_cifar100_class]

        if len(_imagenet_classes) > 0:
            _imagenet_class = random_state.choice(
                _imagenet_classes, size=1, replace=False
            )[0]
            if _imagenet_class not in _selected_imagenet_classes:
                _selected_cifar100_classes.append(_cifar100_class)
                _selected_imagenet_classes.append(_imagenet_class)
            if len(_selected_imagenet_classes) == num_overlap_classes:
                break

    # randomly sample remaining classes
    if num_overlap_classes < num_total_classes:
        matched_classes = functools.reduce(
            lambda a, b: a + b, list(cifar100_class_id_2_imagenet1k_class_id.values())
        )
        unmatched_classes = list(set(range(1000)) - set(matched_classes))
        _selected_imagenet_classes.extend(
            random_state.choice(
                unmatched_classes,
                size=int(num_total_classes - num_overlap_classes),
                replace=False,
            ).tolist()
        )
    return _selected_imagenet_classes


cifar100_class_id_2_imagenet1k_class_id = {
    0: [],
    1: [],
    2: [],
    3: [295, 294, 296, 297],
    4: [337],
    5: [564],
    6: [309],
    7: [302, 301, 304, 303, 300, 307, 587],
    8: [444, 671],
    9: [440, 720, 737, 898, 907],
    10: [659, 809],
    11: [],
    12: [821, 839, 888],
    13: [654, 779, 874],
    14: [326, 322, 325],
    15: [354],
    16: [653],
    17: [483],
    18: [],
    19: [345],
    20: [423, 559, 765],
    21: [367],
    22: [409, 530, 892],
    23: [],
    24: [314],
    25: [],
    26: [118, 120, 121, 119],
    27: [49],
    28: [968],
    29: [],
    30: [148],
    31: [386, 385],
    32: [],
    33: [],
    34: [279, 280, 278, 277],
    35: [],
    36: [333],
    37: [498, 598],
    38: [104],
    39: [508, 878],
    40: [470, 818, 846],
    41: [621],
    42: [288],
    43: [291],
    44: [],
    45: [123],
    46: [],
    47: [],
    48: [],
    49: [970, 980],
    50: [673],
    51: [947, 992, 992],
    52: [],
    53: [950],
    54: [],
    55: [360],
    56: [],
    57: [],
    58: [717],
    59: [],
    60: [],
    61: [923],
    62: [],
    63: [334],
    64: [105],
    65: [332, 330],
    66: [],
    67: [5, 6],
    68: [],
    69: [657],
    70: [],
    71: [],
    72: [],
    73: [4],
    74: [],
    75: [361],
    76: [],
    77: [113],
    78: [65],
    79: [73, 72, 75, 74, 76, 77, 567],
    80: [],
    81: [829],
    82: [],
    83: [945],
    84: [526, 736, 532],
    85: [847, 565],
    86: [528, 707],
    87: [851],
    88: [292],
    89: [866],
    90: [],
    91: [],
    92: [],
    93: [610, 37, 35, 36],
    94: [894],
    95: [],
    96: [],
    97: [272, 271, 269, 270],
    98: [],
    99: [110, 111, 783],
}


cifar100_name_2_imagenet1k_name = {
    "apple": [],
    "aquarium_fish": [],
    "baby": [],
    "bear": [
        "American black bear, black bear, Ursus americanus, Euarctos americanus",
        "brown bear, bruin, Ursus arctos",
        "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
        "sloth bear, Melursus ursinus, Ursus ursinus",
    ],
    "beaver": ["beaver"],
    "bed": ["four-poster"],
    "bee": ["bee"],
    "beetle": [
        "ground beetle, carabid beetle",
        "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
        "leaf beetle, chrysomelid",
        "long-horned beetle, longicorn, longicorn beetle",
        "tiger beetle",
        "weevil",
        "hammer",
    ],
    "bicycle": [
        "bicycle-built-for-two, tandem bicycle, tandem",
        "mountain bike, all-terrain bike, off-roader",
    ],
    "bottle": [
        "beer bottle",
        "pill bottle",
        "pop bottle, soda bottle",
        "water bottle",
        "wine bottle",
    ],
    "bowl": ["mixing bowl", "soup bowl"],
    "boy": [],
    "bridge": ["steel arch bridge", "suspension bridge", "viaduct"],
    "bus": ["minibus", "school bus", "trolleybus, trolley coach, trackless trolley"],
    "butterfly": [
        "lycaenid, lycaenid butterfly",
        "ringlet, ringlet butterfly",
        "sulphur butterfly, sulfur butterfly",
    ],
    "camel": ["Arabian camel, dromedary, Camelus dromedarius"],
    "can": ["milk can"],
    "castle": ["castle"],
    "caterpillar": [],
    "cattle": ["ox"],
    "chair": ["barber chair", "folding chair", "rocking chair, rocker"],
    "chimpanzee": ["chimpanzee, chimp, Pan troglodytes"],
    "clock": ["analog clock", "digital clock", "wall clock"],
    "cloud": [],
    "cockroach": ["cockroach, roach"],
    "couch": [],
    "crab": [
        "Dungeness crab, Cancer magister",
        "fiddler crab",
        "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
        "rock crab, Cancer irroratus",
    ],
    "crocodile": ["African crocodile, Nile crocodile, Crocodylus niloticus"],
    "cup": ["cup"],
    "dinosaur": [],
    "dolphin": ["killer whale, killer, orca, grampus, sea wolf, Orcinus orca"],
    "elephant": [
        "African elephant, Loxodonta africana",
        "Indian elephant, Elephas maximus",
    ],
    "flatfish": [],
    "forest": [],
    "fox": [
        "Arctic fox, white fox, Alopex lagopus",
        "grey fox, gray fox, Urocyon cinereoargenteus",
        "kit fox, Vulpes macrotis",
        "red fox, Vulpes vulpes",
    ],
    "girl": [],
    "hamster": ["hamster"],
    "house": [
        "cinema, movie theater, movie theatre, movie house, picture palace",
        "home theater, home theatre",
    ],
    "kangaroo": ["wallaby, brush kangaroo"],
    "keyboard": ["computer keyboard, keypad", "typewriter keyboard"],
    "lamp": ["candle, taper, wax light", "spotlight, spot", "table lamp"],
    "lawn_mower": ["lawn mower, mower"],
    "leopard": ["leopard, Panthera pardus"],
    "lion": ["lion, king of beasts, Panthera leo"],
    "lizard": [],
    "lobster": [
        "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish"
    ],
    "man": [],
    "maple_tree": [],
    "motorcycle": [],
    "mountain": ["alp", "volcano"],
    "mouse": ["mouse, computer mouse"],
    "mushroom": ["mushroom", "agaric", "agaric"],
    "oak_tree": [],
    "orange": ["orange"],
    "orchid": [],
    "otter": ["otter"],
    "palm_tree": [],
    "pear": [],
    "pickup_truck": ["pickup, pickup truck"],
    "pine_tree": [],
    "plain": [],
    "plate": ["plate"],
    "poppy": [],
    "porcupine": ["porcupine, hedgehog"],
    "possum": ["koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus"],
    "rabbit": ["Angora, Angora rabbit", "wood rabbit, cottontail, cottontail rabbit"],
    "raccoon": [],
    "ray": ["electric ray, crampfish, numbfish, torpedo", "stingray"],
    "road": [],
    "rocket": ["missile"],
    "rose": [],
    "sea": [],
    "seal": [],
    "shark": ["hammerhead, hammerhead shark"],
    "shrew": [],
    "skunk": ["skunk, polecat, wood pussy"],
    "skyscraper": [],
    "snail": ["snail"],
    "snake": ["sea snake"],
    "spider": [
        "barn spider, Araneus cavaticus",
        "black and gold garden spider, Argiope aurantia",
        "black widow, Latrodectus mactans",
        "garden spider, Aranea diademata",
        "tarantula",
        "wolf spider, hunting spider",
        "frying pan, frypan, skillet",
    ],
    "squirrel": [],
    "streetcar": ["streetcar, tram, tramcar, trolley, trolley car"],
    "sunflower": [],
    "sweet_pepper": ["bell pepper"],
    "table": [
        "desk",
        "pool table, billiard table, snooker table",
        "dining table, board",
    ],
    "tank": [
        "tank, army tank, armored combat vehicle, armoured combat vehicle",
        "freight car",
    ],
    "telephone": ["dial telephone, dial phone", "pay-phone, pay-station"],
    "television": ["television, television system"],
    "tiger": ["tiger, Panthera tigris"],
    "tractor": ["tractor"],
    "train": [],
    "trout": [],
    "tulip": [],
    "turtle": [
        "jersey, T-shirt, tee shirt",
        "box turtle, box tortoise",
        "mud turtle",
        "terrapin",
    ],
    "wardrobe": ["wardrobe, closet, press"],
    "whale": [],
    "willow_tree": [],
    "wolf": [
        "coyote, prairie wolf, brush wolf, Canis latrans",
        "red wolf, maned wolf, Canis rufus, Canis niger",
        "timber wolf, grey wolf, gray wolf, Canis lupus",
        "white wolf, Arctic wolf, Canis lupus tundrarum",
    ],
    "woman": [],
    "worm": ["flatworm, platyhelminth", "nematode, nematode worm, roundworm", "screw"],
}
