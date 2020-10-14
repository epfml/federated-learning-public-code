# -*- coding: utf-8 -*-
# some extra parameter parsers

import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return v


def dict_parser(values):
    local_dict = {}
    for kv in values.split(","):
        k, v = kv.split("=")
        try:
            local_dict[k] = float(v)
        except ValueError:
            local_dict[k] = str2bool(v)
        except ValueError:
            local_dict[k] = v
    return local_dict


class DictParser(argparse.Action):
    def __init__(self, *args, **kwargs):

        super(DictParser, self).__init__(*args, **kwargs)
        self.local_dict = {}

    def __call__(self, parser, namespace, values, option_string=None):

        try:
            self.local_dict = dict_parser(values)
            setattr(namespace, self.dest, self.local_dict)
        except:
            raise ValueError("Failed when parsing %s as dict" % values)


class ListParser(argparse.Action):
    def __init__(self, *args, **kwargs):

        super(ListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string=None):

        try:
            self.local_list = values.split(",")
            setattr(namespace, self.dest, self.local_list)
        except:
            raise ValueError("Failed when parsing %s as str list" % values)


class IntListParser(argparse.Action):
    def __init__(self, *args, **kwargs):

        super(IntListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string=None):

        try:
            self.local_list = list(map(int, values.split(",")))
            setattr(namespace, self.dest, self.local_list)
        except:
            raise ValueError("Failed when parsing %s as int list" % values)


class FloatListParser(argparse.Action):
    def __init__(self, *args, **kwargs):

        super(FloatListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string=None):

        try:
            self.local_list = list(map(float, values.split(",")))
            setattr(namespace, self.dest, self.local_list)
        except:
            raise ValueError("Failed when parsing %s as float list" % values)


class BooleanParser(argparse.Action):
    def __init__(self, *args, **kwargs):

        super(BooleanParser, self).__init__(*args, **kwargs)
        self.values = None

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            self.values = False if int(values) == 0 else True
            setattr(namespace, self.dest, self.values)
        except:
            raise ValueError("Failed when parsing %s as boolean list" % values)
