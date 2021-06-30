from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from .bert_formatting import (
    glue_example_to_feature,
    tagging_example_to_feature,
    multiplechoice_example_to_feature,
)
from .glue.datasets import *
from .textcls.text_classification import *
import torch
import os
import pickle
import json
import uuid
import numpy as np


task2datadir = {
    "mrpc": "data/glue_data/data/MRPC",
    "sst2": "data/glue_data/data/SST-2",
    "mnli": "data/glue_data/data/MNLI",
    "qqp": "data/glue_data/data/QQP",
    "cola": "data/glue_data/data/CoLA",
    "qnli": "data/glue_data/data/QNLI",
    "rte": "data/glue_data/data/RTE",
    "posptb": "data/tagging/ptb/",
    "swag": "data/reading_compr/swag",
    "squad1": "data/qa/squad/v1.1",
    "agnews": "data/textcls/agnews",
    "trec": "data/textcls/trec",
    "dbpedia": "data/textcls/dbpedia",
    "yelp2": "data/textcls/yelp2",
    "semeval16": "data/semeval/",
    "conll2003": "data/ner/conll2003",
}


task2dataset = {
    "mrpc": MRPCDataset,
    "sst2": SST2Dataset,
    "mnli": MNLIDataset,
    "qqp": QQPDataset,
    "cola": COLADataset,
    "qnli": QNLIDataset,
    "rte": RTEDataset,
    "agnews": AGNEWSDataset,
    "trec": TRECDataset,
    "dbpedia": DBPEDIADataset,
    "yelp2": YELPDataset,
}


task2metrics = {
    "mrpc": ["accuracy"],
    "sst2": ["accuracy"],
    "mnli": ["accuracy"],
    "qqp": ["f1", "accuracy"],
    "cola": ["mcc"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "posptb": ["accuracy"],
    "swag": ["accuracy"],
    "agnews": ["accuracy"],
    "trec": ["accuracy"],
    "dbpedia": ["accuracy"],
    "yelp2": ["accuracy"],
    "semeval16": ["accuracy"],
    "conll2003": ["f1_score_ner"],
}


class SeqClsDataIter(object):
    def __init__(self, task, model, tokenizer, max_seq_len, conf):
        self.conf = conf
        self.task = task
        self.metrics = task2metrics[task]
        self.pdata = task2dataset[task](task2datadir[task])
        self.num_labels = len(self.pdata.get_labels())
        self.trn_dl = self.wrap_iter(
            task, model, "trn", self.pdata.trn_egs, tokenizer, max_seq_len
        )
        self.val_dl = self.wrap_iter(
            task, model, "val", self.pdata.val_egs, tokenizer, max_seq_len
        )
        if hasattr(self.pdata, "tst_egs"):
            self.tst_dl = self.wrap_iter(
                task, model, "tst", self.pdata.tst_egs, tokenizer, max_seq_len
            )

    def wrap_iter(self, task, model, split, egs, tokenizer, max_seq_len):
        cached_ = os.path.join(
            "data", "cached", f"{task},{max_seq_len},{model},{split},cached.pkl"
        )
        meta_ = cached_.replace(".pkl", ".meta")
        if os.path.exists(cached_):
            print("[INFO] loading cached dataset.")
            with open(meta_, "r") as f:
                meta = json.load(f)
            assert meta["complete"]
            with open(cached_, "rb") as f:
                fts = pickle.load(f)
            if fts["uid"] == meta["uid"]:
                fts = fts["fts"]
            else:
                # will not do self recompute for safety
                raise ValueError("uids of data and meta do not match ...")
        else:
            print("[INFO] computing fresh dataset.")
            fts = glue_example_to_feature(
                self.task, egs, tokenizer, max_seq_len, self.label_list
            )
            uid, complete = str(uuid.uuid4()), True
            try:
                with open(cached_, "wb") as f:
                    pickle.dump({"fts": fts, "uid": uid}, f)
            except:
                complete = False
            with open(meta_, "w") as f:
                json.dump({"complete": complete, "uid": uid}, f)
        return _SeqClsIter(fts, conf=self.conf)

    @property
    def name(self):
        return self.pdata.name

    @property
    def label_list(self):
        return self.pdata.get_labels()


class _SeqClsIter(torch.utils.data.Dataset):
    def __init__(self, fts, conf, shuffle=True):
        ind_map = np.array([x for x in range(0, len(fts))])
        if shuffle:
            conf.random_state.shuffle(ind_map)

        self.uides = [fts[x].uid for x in ind_map]
        self.input_idses = torch.as_tensor(
            [fts[x].input_ids for x in ind_map], dtype=torch.long
        )
        self.golds = torch.as_tensor([fts[x].gold for x in ind_map], dtype=torch.long)
        self.attention_maskes = torch.as_tensor(
            [fts[x].attention_mask for x in ind_map], dtype=torch.long
        )
        self.token_type_idses = torch.as_tensor(
            [fts[x].token_type_ids for x in ind_map], dtype=torch.long
        )

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.golds[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )
