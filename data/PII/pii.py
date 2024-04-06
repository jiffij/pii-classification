import json
# import argparse
from itertools import chain
# from functools import partial
# import evaluate
from transformers import AutoTokenizer#, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import os
import torch
import random

TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
TRAINING_MAX_LENGTH = 1024
OUTPUT_DIR = "output"
PII_MODES = ["TRAIN", "VALID", "TEST"]


class PII(object):
    def __init__(self, path, num_valid_samples = 200):
        self.mode = PII_MODES[0]
        self.tokenizer, self.train, self.dictionary = self.tokenize(path)
        self.valid_idx = random.sample(range(len(self.train)), num_valid_samples)
        # self.valid = self.train[self.valid_idx]
        
        indices_to_keep = [idx for idx in range(len(self.train)) if idx not in self.valid_idx]
        # Create a new dataset with the remaining items
        
        self.valid = self.train.select(self.valid_idx)
        self.train = self.train.select(indices_to_keep)
        # self.valid = self.train.filter(lambda example, idx: idx not in indices_to_keep)
        # self.train = self.train.filter(lambda example, idx: idx not in self.valid_idx)
        


    def __getitem__(self, items): 
        if self.mode == PII_MODES[0]:
            x = self.train[items]
        elif self.mode == PII_MODES[1]:
            x = self.valid[items]
            pass
        else:
            return None
        return torch.LongTensor(x["input_ids"]), torch.LongTensor(x['labels'])
    
    def __len__(self):
        if self.mode == PII_MODES[0]:
            return len(self.train)
        elif self.mode == PII_MODES[1]:
            return len(self.valid)
        else: 
            return -1
    
    def tokenize(self, path):
        data = json.load(open(os.path.join(path, "pii-detection-removal-from-educational-data", "train.json")))

        # downsampling of negative examples
        p=[] # positive samples (contain relevant labels)
        n=[] # negative samples (presumably contain entities that are possibly wrongly classified as entity)
        for d in data:
            if any(np.array(d["labels"]) != "O"): p.append(d)
            else: n.append(d)
        print("original datapoints: ", len(data))

        external = json.load(open(os.path.join(path, "mixtral-8x7b-v1", "mixtral-8x7b-v1.json")))
        print("external datapoints: ", len(external))

        # moredata = json.load(open("/kaggle/input/fix-punctuation-tokenization-external-dataset/moredata_dataset_fixed.json"))
        # print("moredata datapoints: ", len(moredata))

        data = external+p+n[:len(n)//3] #moredata+
        print("combined: ", len(data))



        all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
        label2id = {l: i for i,l in enumerate(all_labels)}
        id2label = {v:k for k,v in label2id.items()}

        target = [
            'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
            'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
            'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
        ]

        print(id2label)


        def tokenize(example, tokenizer, label2id, max_length):

            # rebuild text from tokens
            text = []
            labels = []

            for t, l, ws in zip(
                example["tokens"], example["provided_labels"], example["trailing_whitespace"]
            ):
                text.append(t)
                labels.extend([l] * len(t))

                if ws:
                    text.append(" ")
                    labels.append("O")

            # actual tokenization
            tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

            labels = np.array(labels)

            text = "".join(text)
            token_labels = []

            for start_idx, end_idx in tokenized.offset_mapping:
                # CLS token
                if start_idx == 0 and end_idx == 0:
                    token_labels.append(label2id["O"])
                    continue

                # case when token starts with whitespace
                if text[start_idx].isspace():
                    start_idx += 1

                token_labels.append(label2id[labels[start_idx]])

            length = len(tokenized.input_ids)

            return {**tokenized, "labels": token_labels, "length": length}


        tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

        ds = Dataset.from_dict({
            "full_text": [x["full_text"] for x in data],
            "document": [str(x["document"]) for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        })
        ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH}, num_proc=3)

        x = ds[0]

        for t,l in zip(x["tokens"], x["provided_labels"]):
            if l != "O":
                print((t,l))

        print("*"*100)

        for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
            if id2label[l] != "O":
                print((t,id2label[l]))
                
        # print("*"*100)
        
        # print(x["input_ids"], x['labels'])
                
        return tokenizer, ds, id2label
    

# if __name__ == "__main__":
#     pii = PII(os.path.join('..', 'dataset'))
#     print(len(pii))
#     print(pii[1])
#     pii.mode = PII_MODES[1]
#     print(len(pii))
#     print(pii[1])