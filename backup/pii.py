import json
# import argparse
from itertools import chain
# from functools import partial
# import evaluate
from transformers import AutoTokenizer  # , Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import os
import torch
import random
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import FastText

# from torchtext.vocab import Vocab, build_vocab_from_iterator
# from collections import Counter, OrderedDict

TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
TRAINING_MAX_LENGTH = 1024
OUTPUT_DIR = "output"
PII_MODES = ["TRAIN", "VALID", "TEST"]
random.seed(10)


# class CustomVocab(torchtext.vocab.Vocab):
#     def set_vectors(self, vectors):
#         self.vectors = vectors

class PII(object):
    def __init__(self, path, num_valid_samples=0.1, use_cuda=False, cal_class_weight=False, build_vocab=False,
                 fasttext_file="pii_fast_min2.ft", use_fast=True, language_model=False):
        self.mode = PII_MODES[0]
        self.language_model = language_model
        self.use_cuda = use_cuda
        self.tokenizer, self.train, self.id2label = self.tokenize(path)
        self.num_classes = len(self.id2label)
        self.use_fast = use_fast
        self.vocab_dim = 1024 if use_fast else self.tokenizer.vocab_size
        if cal_class_weight:
            self.class_weight = self.get_class_weight(self.train)
        else:
            self.class_weight = torch.tensor([7.1938e+00, 2.0920e+01, 1.1170e+01, 3.0067e+01, 3.8373e+01, 5.8495e+00,
                2.6150e+01, 7.8843e+01, 2.3592e+01, 1.3449e+01, 8.9839e+00, 2.9242e+04,
                8.1871e-02]) # TODO change 02 to 03


        if build_vocab:
            tokenized_sentences = [self.tokenizer.convert_ids_to_tokens(sentence['input_ids']) for sentence in
                                   self.train]
            fast = FastText(sentences=tokenized_sentences, vector_size=1024, window=5, min_count=3, workers=4,
                            epochs=10, sg=1)
            word_embeddings = fast.wv
            print("Corpus total words:", fast.corpus_total_words)
            print(f"Fast text Embedding size: {fast.wv.vector_size}")
            fast.save(fasttext_file)
            print(f"Model saved as {fasttext_file}...")

        if use_fast:
            self.encoder = FastText.load(fasttext_file)
            print("loaded encoder..")

        # print(word_embeddings['the'])
        # print(word_embeddings.similarity("▁Richard", "▁brandy"))
        # c = []
        # for i in range(len(self.train)):
        #     c = c + self.tokenizer.convert_ids_to_tokens(self.train[i]["input_ids"])
        # counter = Counter(c)
        # sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # unk_token = '<unk>'
        # default_index = -1
        # ordered_dict = OrderedDict(sorted_by_freq_tuples, specials=[unk_token])
        # v1 = Vocab(ordered_dict)
        # # v1.set_default_index(default_index)
        # v1.set_default_index(v1[unk_token])
        # self.vocab = v1

        # unk_token = '<unk>'
        # self.vocab = build_vocab_from_iterator(self.yield_tokens(self.train), specials=[unk_token])
        # self.vocab.set_default_index(self.vocab[unk_token])

        # # Initialize a custom vocabulary
        # self.vocab = torchtext.vocab.Vocab(max_size=50000)

        # Build the vocabulary
        # self.vocab.build_vocab_from_iterator(self.custom_iterator(self.train))

        # with everything: (total 11188 samples)
        # [7.1938e+00, 2.0920e+01, 1.1170e+01, 3.0067e+01, 3.8373e+01, 5.8495e+00,
        # 2.6150e+01, 7.8843e+01, 2.3592e+01, 1.3449e+01, 8.9839e+00, 2.9242e+04,
        # 8.1871e-02]
        # log:
        # torch.tensor([7.45725772, 8.99731462, 8.09206095, 9.52061269, 9.87252333,
        #               7.15881698, 9.31924271, 10.91141444, 9.1707295, 8.35993067,
        #               7.77784554, 19.44625804, 1.])

        # only mixtral:
        # log weight:
        # torch.tensor([ 8.56314318,  9.19172856,  9.5906706 , 10.30031814, 10.98184035,
        # 7.23594694,  9.85339371, 11.1952026 ,  9.53953753,  9.14359358,
        # 8.48692677, 20.09521001,  1.        ])
        # self.class_weight = torch.tensor([1.5158e+01, 2.3435e+01, 3.0900e+01, 5.0534e+01, 8.1048e+01, 6.0411e+00,
        # 3.7072e+01, 9.3966e+01, 2.9824e+01, 2.2666e+01, 1.4378e+01, 4.4889e+04,
        # 8.0151e-02], dtype=torch.float)

        # self.dictionary_size = len(self.train.features.keys())
        # print(self.tokenizer.vocab_size)
        self.valid_idx = random.sample(range(len(self.train)), round(len(self.train) * num_valid_samples))
        # self.valid = self.train[self.valid_idx]

        indices_to_keep = [idx for idx in range(len(self.train)) if idx not in self.valid_idx]
        # Create a new dataset with the remaining items

        self.valid = self.train.select(self.valid_idx)
        self.train = self.train.select(indices_to_keep)
        # self.valid = self.train.filter(lambda example, idx: idx not in indices_to_keep)
        # self.train = self.train.filter(lambda example, idx: idx not in self.valid_idx)

    # Define a custom iterator that yields tokenized inputs
    def yield_tokens(self, DS):
        for t in DS:
            yield self.tokenizer.convert_ids_to_tokens(t["input_ids"])

    def __getitem__(self, items):
        if self.mode == PII_MODES[0]:
            x = self.train[items]
        elif self.mode == PII_MODES[1]:
            x = self.valid[items]
            pass
        else:
            return None

        if self.language_model:
            X, y = torch.tensor(
                np.array([self.encoder.wv[w] for w in self.tokenizer.convert_ids_to_tokens(x['input_ids'])]),
                dtype=torch.float32)[:-1], \
                torch.tensor(np.array([self.encoder.wv[w] for w in self.tokenizer.convert_ids_to_tokens(x['input_ids'])]),
                                dtype=torch.float32)[1:]
        elif self.use_fast:
            X, y = torch.tensor(np.array([self.encoder.wv[w] for w in self.tokenizer.convert_ids_to_tokens(x['input_ids'])]),
                                dtype=torch.float32), torch.LongTensor(x['labels'])
        else:
            X, y = torch.LongTensor(x["input_ids"]), torch.LongTensor(x['labels'])

        if self.use_cuda:
            return X.cuda(), y.cuda()
        else:
            return X, y  # torch.nn.functional.one_hot(torch.tensor(x['labels']), num_classes=self.num_classes)

    def __len__(self):
        if self.mode == PII_MODES[0]:
            return len(self.train)
        elif self.mode == PII_MODES[1]:
            return len(self.valid)
        else:
            return -1

    def get_class_weight(self, df):
        y = []
        for i in range(len(df)):
            y = y + df[i]['labels']
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=np.array(y))
        return torch.tensor(class_weights, dtype=torch.float32)

    def tokenize(self, path):
        data = json.load(open(os.path.join(path, "pii-detection-removal-from-educational-data", "train.json")))

        # downsampling of negative examples
        p = []  # positive samples (contain relevant labels)
        n = []  # negative samples (presumably contain entities that are possibly wrongly classified as entity)
        for d in data:
            if any(np.array(d["labels"]) != "O"):
                p.append(d)
            else:
                n.append(d)
        print("original datapoints: ", len(data))

        mixtral = json.load(open(os.path.join(path, "mixtral-8x7b-v1", "mixtral-8x7b-v1.json")))
        print("mixtral datapoints: ", len(mixtral))

        moredata = json.load(open(os.path.join(path, "moredata_dataset_fixed.json")))
        print("moredata datapoints: ", len(moredata))

        external = json.load(open(os.path.join(path, "pii_dataset_fixed.json")))
        print("external datapoints: ", len(external))

        data = mixtral + moredata + external + p + n[:len(n) // 3]  # moredata+
        print("combined: ", len(data))

        all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
        label2id = {l: i for i, l in enumerate(all_labels)}
        id2label = {v: k for k, v in label2id.items()}

        target = [
            'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
            'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
            'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
        ]

        print(id2label)
        if self.language_model:
            pad = 1
        else:
            pad = 0
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
            tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length + pad, truncation=True,
                                  padding=True)

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
        training_max_length = TRAINING_MAX_LENGTH + pad
        ds = ds.map(tokenize,
                    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": training_max_length},
                    num_proc=3)
        ds = ds.shuffle(seed=10)
        # x = ds[0]

        # for t,l in zip(x["tokens"], x["provided_labels"]):
        #     if l != "O":
        #         print((t,l))

        # print("*"*100)

        # for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
        #     if id2label[l] != "O":
        #         print((t,id2label[l]))

        # print("*"*100)

        # print(x["input_ids"], x['labels'])

        return tokenizer, ds, id2label


# if __name__ == "__main__":
#     pii = PII(os.path.join('..', 'dataset'), cal_class_weight=False, build_vocab=True)
#     # print(pii.class_weight)
#     print(len(pii.vocab))
#     print(pii.vocab.itos[:10])

#     print(len(pii))
#     print(pii[1])
# pii.mode = PII_MODES[1]
# print(len(pii))
# print(pii[1])
