import json
from lib2to3.pgen2.tokenize import tokenize
from click import option
import torch
import os
import numpy as np

import random
from transformers import BertTokenizer,AutoTokenizer
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForLanguageModeling
from tools import print_rank

class SQuADFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.config = config
        print("formatter", config.get("model", "pretrained_model"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model", "pretrained_model"))
        self.question_first = config.getboolean("train", "question_first")

    def process(self, data):

        allquestions = [d["question"].strip() for d in data]
        allctxs = [d["context"] for d in data]
        allanswer = [d["answers"] for d in data]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            allquestions if self.question_first else allctxs,
            allctxs if self.question_first else allquestions,
            truncation="only_second" if self.question_first else "only_first",
            max_length=self.max_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        if self.mode == "train":
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = allanswer[sample_index]
                # If no answers are given, set the cls_index as answer.
                if len(answers) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers[0]["answer_start"]
                    end_char = start_char + len(answers[0]["text"])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if self.question_first else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if self.question_first else 0):
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)
        else:
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if self.question_first else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(data[sample_index]["id"])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            tokenized_examples["oridata"] = data

        for key in tokenized_examples:
            if key not in ["example_id", "oridata"]:
                tokenized_examples[key] = torch.LongTensor(tokenized_examples[key])

        return tokenized_examples