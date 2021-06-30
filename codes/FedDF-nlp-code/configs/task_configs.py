from data_loader import SeqClsDataIter
from transformers.tokenization_bert import BertTokenizer


task2main_metric = {
    "mrpc": "accuracy",
    "sst2": "accuracy",
    "mnli": "accuracy",
    "qqp": "f1",
    "cola": "mcc",
    "qnli": "accuracy",
    "rte": "accuracy",
    "posptb": "accuracy",
    "swag": "accuracy",
    "agnews": "accuracy",
    "trec": "accuracy",
    "dbpedia": "accuracy",
    "yelp2": "accuracy",
    "semeval16": "accuracy",
    "conll2003": "f1_score_ner",
}


task2dataiter = {
    "mrpc": SeqClsDataIter,
    "sst2": SeqClsDataIter,
    "mnli": SeqClsDataIter,
    "qqp": SeqClsDataIter,
    "cola": SeqClsDataIter,
    "qnli": SeqClsDataIter,
    "rte": SeqClsDataIter,
    "agnews": SeqClsDataIter,
    "trec": SeqClsDataIter,
    "dbpedia": SeqClsDataIter,
    "yelp2": SeqClsDataIter,
    "semeval16": SeqClsDataIter,
}
