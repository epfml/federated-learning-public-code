import os
import csv

from ..common import MultipleChoiceExample

__all__ = ["SWAGDataset"]


class MultipleChoiceDataset(object):
    def get_split_examples(self):
        raise NotImplementedError

    def _create_examples(self):
        raise NotImplementedError


class SWAGDataset(MultipleChoiceDataset):
    def __init__(self, data_dir):
        self.name = "swag"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("val")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.csv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def _create_examples(self, where_, which_split):
        with open(where_, "r") as f:
            lines = list(csv.reader(f))[1:]
        examples = [
            MultipleChoiceExample(
                uid=line[2],
                context=line[4],
                start_choice=line[5],
                choices=line[7:11],
                label=int(line[11]),
            )
            for line in lines
        ]
        return examples
