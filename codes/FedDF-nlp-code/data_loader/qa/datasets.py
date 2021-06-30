import os
import json
import tqdm


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


class SquadExample(object):
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.answers = answers
        self.is_impossible = is_impossible

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed
        # to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(
                    start_position_character + len(answer_text) - 1,
                    len(char_to_word_offset) - 1,
                )
            ]


class SQUADDataset(object):
    def __init__(self, version, data_dir):
        assert version[0] == "v"
        self.name = "squad-{}".format(version)
        self.version = version[1:]
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.json".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def _create_examples(self, where_, which_split):
        with open(where_, "r") as f:
            input_json = json.load(f)
        assert self.version == input_json["version"].replace("v", "")
        data = input_json["data"]
        examples = []
        for entry in tqdm.tqdm(data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []
                    if "is_impossible" in qa:  # only useful for v2.0
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False
                    if not is_impossible:
                        if which_split == "train":
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples
