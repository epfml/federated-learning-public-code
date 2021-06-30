import os
from ..common import SentencePairExample, SentenceExample


__all__ = [
    "MRPCDataset",
    "SST2Dataset",
    "MNLIDataset",
    "QQPDataset",
    "COLADataset",
    "QNLIDataset",
    "RTEDataset",
]


class GlueDataset(object):
    def get_split_examples(self, split):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def _create_examples(self):
        raise NotImplementedError


class MRPCDataset(GlueDataset):
    def __init__(self, dara_dir):
        self.name = "mrpc"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 5
                label = segs[0]
                text_a, text_b = segs[3], segs[4]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs


class QNLIDataset(GlueDataset):
    """ a sentence pair dataset converted from squad. """

    def __init__(self, dara_dir):
        self.name = "qnli"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 4
                label = segs[-1]
                text_a, text_b = segs[1], segs[2]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs


class RTEDataset(GlueDataset):
    """ a sentence pair dataset converted from squad. """

    def __init__(self, dara_dir):
        self.name = "rte"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 4
                label = segs[-1]
                text_a, text_b = segs[1], segs[2]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs


class MNLIDataset(GlueDataset):
    """ 
    multi-genre NLI: 
        for a pair of sentences, predict whether the second sentence is an entailment, 
        contradiction, or neutral w.r.t the first one. 
    NB: 
        this dataset seems to be problematic.
        BERT only use MNLI-m, in domain classification.
    """

    def __init__(self, data_dir):
        self.name = "mnli"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev_matched")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentenceExample """
        sentence_pair_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                uid = "%s-%s" % (which_split, idx)
                text_a, text_b = line[8], line[9]
                label = line[-1]
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs


class QQPDataset(GlueDataset):
    """
    Determine whether a pair of questions are semantically equivalent.
    """

    def __init__(self, data_dir):
        self.name = "qqp"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentenceExample 
        there are some ill examples in this dataset so they are skipped.
        """
        sentence_pair_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                uid = "%s-%s" % (which_split, idx)
                try:
                    text_a, text_b, label = line[3], line[4], line[5]
                except IndexError:  # it seems transformers also did this
                    print(idx, line)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs


class SST2Dataset(GlueDataset):
    def __init__(self, data_dir):
        self.name = "sst2"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentenceExample """
        sentence_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                assert len(line) == 2
                text_a, label = line
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs


class COLADataset(GlueDataset):
    def __init__(self, data_dir):
        self.name = "cola"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentenceExample """
        sentence_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                text_a, label = line[3], line[1]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs
