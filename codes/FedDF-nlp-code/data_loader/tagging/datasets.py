import os
from nltk.tree import Tree
from tqdm import tqdm


__all__ = ["PTBTDataset"]


class PTBTDataset(object):
    def __init__(self, data_dir):
        self.name = "ptbtagging"
        self.data_dir = data_dir
        trn_secs = ["0" + y if int(y) <= 9 else y for y in [str(x) for x in range(19)]]
        val_secs = ["19", "20", "21"]

        self.trn_tagged_sents = self.parse_split(trn_secs)
        self.val_tagged_sents = self.parse_split(val_secs)

        all_tags = set()
        for tagged_sent in self.trn_tagged_sents + self.val_tagged_sents:
            for word, tag in tagged_sent:
                all_tags.add(tag)
        all_tags = sorted(list(all_tags))  # determinstic tag <-> idx mapping
        t2i = {t: i for i, t in enumerate(all_tags)}
        t2i["<PAD>"] = len(t2i)
        self.t2i = t2i

    def get_labels(self):
        return self.t2i.keys()

    def parse_split(self, secs):
        print("[INFO] parsing the tagged trees ...")
        tagged_sents = []
        for sec in tqdm(secs):
            tagged_sents.extend(self._parse_trees_in_section(sec))
        return tagged_sents

    def _parse_trees_in_section(self, sec):
        tagged_sents = []
        sec_dir = os.path.join(self.data_dir, sec)
        for _tree in os.listdir(sec_dir):
            tree = os.path.join(sec_dir, _tree)
            if not _tree.startswith(r"."):
                with open(tree, "r", encoding="utf-8") as f:
                    for line in f:
                        word_tag_pairs = Tree.fromstring(line).pos()
                        tagged_sents.append(word_tag_pairs)
        return tagged_sents
