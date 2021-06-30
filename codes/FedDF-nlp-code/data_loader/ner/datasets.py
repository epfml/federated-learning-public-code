import os
from collections import defaultdict

__all__ = ["CONLL2003Dataset"]


class CONLL2003Dataset(object):
    """ @zhao: i will just follow the transformer package to make ner as a tagging
    task, though im not very convinced that this treatment would make sense ...
    - testa is the dev; testb is the actual test.
    - "O" is a proper label being used in loss computation.
    """

    def __init__(self, data_dir):
        self.name = "conll2003"
        self.data_dir = data_dir
        self.tagset = None
        self.trn_tagged_sents = self.parse_split(os.path.join(data_dir, "eng.train"))
        self.val_tagged_sents = self.parse_split(os.path.join(data_dir, "eng.testa"))
        # self.tst_tagged_sents = self.parse_split(os.path.join(data_dir, "eng.testb"))

        tagset = set()
        for sent in self.trn_tagged_sents + self.val_tagged_sents:
            for _, tag in sent:
                tagset.add(tag)
        tagset = sorted(list(tagset))
        t2i = {t: i for i, t in enumerate(tagset)}
        t2i["<PAD>"] = len(t2i)
        self.pad_id = t2i["<PAD>"]
        self.t2i = t2i
        self.i2t = {i: t for t, i in self.t2i.items()}

    def get_labels(self):
        return self.t2i.keys()

    def parse_split(self, split_):
        # or just use nltk
        sents = []
        with open(split_, "r") as f:
            words, labels = [], []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        assert len(words) == len(labels)
                        sents.append(list(zip(words, labels)))
                        words, labels = [], []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        labels.append("O")
            if words:
                sents.append(zip(words, labels))
        return sents


if __name__ == "__main__":
    ds = CONLL2003Dataset("/mounts/work/mzhao/berttyback/code/data/ner/conll2003")
