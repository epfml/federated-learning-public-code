import copy
import json


class SentenceExample(object):
    def __init__(self, uid, text_a, label=None):
        self.uid = uid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentencePairExample(SentenceExample):
    def __init__(self, uid, text_a, text_b=None, label=None):
        super(SentencePairExample, self).__init__(uid, text_a, label)
        self.text_b = text_b


class MultipleChoiceExample(object):
    def __init__(self, uid, context, start_choice, choices, label=None):
        self.uid = uid
        self.context = context
        self.start_choice = start_choice
        self.choices = choices
        self.label = label
        self.num_choices = len(self.choices)

    def __repr__(self):
        return json.dumps(
            {
                "uid": self.uid,
                "context": self.context,
                "start_choice": self.start_choice,
                "num_choices": self.num_choices,
                "choices": self.choices,
                "label": self.label,
            }
        )
