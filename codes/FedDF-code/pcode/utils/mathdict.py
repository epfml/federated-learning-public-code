class MathDict:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.keys = set(dictionary.keys())

    def __str__(self):
        return "MathDict({})".format(str(self.dictionary))

    def __repr__(self):
        return "MathDict({})".format(repr(self.dictionary))

    def map(self, mapfun):
        new_dict = {}
        for key in self.keys:
            new_dict[key] = mapfun(self.dictionary[key])
        return MathDict(new_dict)

    def filter(self, condfun):
        new_dict = {}
        for key in self.keys:
            if condfun(key):
                new_dict[key] = self.dictionary[key]
        return MathDict(new_dict)

    def detach(self):
        for key in self.keys:
            self.dictionary[key] = self.dictionary[key].detach()

    def values(self):
        return self.dictionary.values()

    def items(self):
        return self.dictionary.items()


def _mathdict_binary_op(operation):
    def op(self, other):
        new_dict = {}
        if isinstance(other, MathDict):
            assert other.keys == self.keys
            for key in self.keys:
                new_dict[key] = operation(self.dictionary[key], other.dictionary[key])
        else:
            for key in self.keys:
                new_dict[key] = operation(self.dictionary[key], other)
        return MathDict(new_dict)

    return op


def _mathdict_map_op(operation):
    def op(self, *args, **kwargs):
        new_dict = {}
        for key in self.keys:
            new_dict[key] = operation(self.dictionary[key], args, kwargs)
        return MathDict(new_dict)

    return op


def _mathdict_binary_in_place_op(operation):
    def op(self, other):
        if isinstance(other, MathDict):
            assert other.keys == self.keys
            for key in self.keys:
                operation(self.dictionary, key, other.dictionary[key])
        else:
            for key in self.keys:
                operation(self.dictionary, key, other)
        return self

    return op


def _iadd(dict, key, b):
    dict[key] += b


def _isub(dict, key, b):
    dict[key] -= b


def _imul(dict, key, b):
    dict[key] *= b


def _itruediv(dict, key, b):
    dict[key] /= b


def _ifloordiv(dict, key, b):
    dict[key] //= b


MathDict.__add__ = _mathdict_binary_op(lambda a, b: a + b)
MathDict.__sub__ = _mathdict_binary_op(lambda a, b: a - b)
MathDict.__rsub__ = _mathdict_binary_op(lambda a, b: b - a)
MathDict.__mul__ = _mathdict_binary_op(lambda a, b: a * b)
MathDict.__rmul__ = _mathdict_binary_op(lambda a, b: a * b)
MathDict.__truediv__ = _mathdict_binary_op(lambda a, b: a / b)
MathDict.__floordiv__ = _mathdict_binary_op(lambda a, b: a // b)
MathDict.__getitem__ = _mathdict_map_op(
    lambda x, args, kwargs: x.__getitem__(*args, **kwargs)
)
MathDict.__iadd__ = _mathdict_binary_in_place_op(_iadd)
MathDict.__isub__ = _mathdict_binary_in_place_op(_isub)
MathDict.__imul__ = _mathdict_binary_in_place_op(_imul)
MathDict.__itruediv__ = _mathdict_binary_in_place_op(_itruediv)
MathDict.__ifloordiv__ = _mathdict_binary_in_place_op(_ifloordiv)
