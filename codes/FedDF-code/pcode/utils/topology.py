# -*- coding: utf-8 -*-
import functools


def configure_gpu(world_conf):
    # the logic of world_conf follows "a,b,c,d,e" where:
    # the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time);
    # the block will be repeated for 'e' times.
    start, stop, interval, local_repeat, block_repeat = [
        int(x) for x in world_conf.split(",")
    ]
    _block = [
        [x] * local_repeat for x in range(start, stop + 1, interval)
    ] * block_repeat
    world_list = functools.reduce(lambda a, b: a + b, _block)
    return world_list


class PhysicalLayout(object):
    def __init__(self, n_participated, world, world_conf, on_cuda):
        self.n_participated = n_participated
        self._world = self.configure_world(world, world_conf)
        self._on_cuda = on_cuda
        self.rank = -1

    def configure_world(self, world, world_conf):
        if world is not None:
            world_list = world.split(",")
            assert self.n_participated <= len(world_list)
        elif world_conf is not None:
            # the logic of world_conf follows "a,b,c,d,e" where:
            # the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time);
            # the block will be repeated for 'e' times.
            return configure_gpu(world_conf)
        else:
            raise RuntimeError(
                "you should at least make sure world or world_conf is not None."
            )
        return [int(l) for l in world_list]

    @property
    def primary_device(self):
        return self.devices[0]

    @property
    def devices(self):
        return self.world

    @property
    def on_cuda(self):
        return self._on_cuda

    @property
    def ranks(self):
        return list(range(1 + self.n_participated))

    @property
    def world(self):
        return self._world

    def get_device(self, rank):
        return self.devices[rank]

    def change_n_participated(self, n_participated):
        self.n_participated = n_participated


def define_graph_topology(world, world_conf, n_participated, on_cuda):
    return PhysicalLayout(
        n_participated=n_participated,
        world=world,
        world_conf=world_conf,
        on_cuda=on_cuda,
    )
