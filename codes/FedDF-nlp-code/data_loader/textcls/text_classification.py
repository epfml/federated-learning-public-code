import os
from ..glue.datasets import GlueDataset
from ..common import SentenceExample


__all__ = ["AGNEWSDataset", "TRECDataset", "DBPEDIADataset", "YELPDataset"]


class AGNEWSDataset(GlueDataset):
    def __init__(self, dara_dir):
        self.name = "agnews"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train_split")
        self.val_egs = self.get_split_examples("dev_split")
        self.tst_egs = self.get_split_examples("test_split")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["1", "2", "3", "4"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                label = line[0]
                assert int(label) in [1, 2, 3, 4]
                text_a = line[1]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs


class TRECDataset(GlueDataset):
    def __init__(self, dara_dir):
        self.name = "trec"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train_split")
        self.val_egs = self.get_split_examples("dev_split")
        self.tst_egs = self.get_split_examples("test_split")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                label = line[0]
                text_a = line[1]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs


class DBPEDIADataset(GlueDataset):
    def __init__(self, dara_dir):
        self.name = "dbpedia"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train_split")
        self.val_egs = self.get_split_examples("dev_split")
        self.tst_egs = self.get_split_examples("test_split")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return [str(x) for x in range(1, 15)]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                label = line[0]
                text_a = line[1]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs


class YELPDataset(GlueDataset):
    def __init__(self, dara_dir):
        self.name = "yelp2"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train_split")
        self.val_egs = self.get_split_examples("dev_split")
        # self.tst_egs = self.get_split_examples("test_split")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                label = line[0]
                text_a = line[1]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs
