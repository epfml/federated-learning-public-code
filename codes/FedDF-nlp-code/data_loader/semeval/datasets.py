import os
from ..glue.datasets import GlueDataset
from ..common import SentenceExample


__all__ = ["SEMEVAL16Dataset"]


class SEMEVAL16Dataset(GlueDataset):
    def __init__(self, dara_dir):
        self.name = "semeval16"
        self.data_dir = dara_dir
        self.trn_egs = self.get_split_examples("train_split")
        self.val_egs = self.get_split_examples("dev_split")
        self.tst_egs = self.get_split_examples("test_split")

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
                label = line[1]
                assert int(label) in [0, 1]
                text_a = line[0]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs
