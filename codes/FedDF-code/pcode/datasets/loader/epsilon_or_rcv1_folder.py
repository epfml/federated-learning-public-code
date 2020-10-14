# -*- coding: utf-8 -*-

from pcode.datasets.loader.utils import LMDBPT


def define_epsilon_or_rcv1_folder(root):
    print("load epsilon_or_rcv1 from lmdb: {}.".format(root))
    return LMDBPT(root, is_image=False)
