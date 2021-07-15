#!/usr/bin/env python
# coding: utf-8

import codecs
import csv
import json
import logging
import os
import re
import sys
import time
from itertools import product

import dill
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
#from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def clean_text(string):
    output = string.strip()
    # replacements = (("“", '"'), ("”", '"'), ("//", ""), ("«", '"'), ("»",'"'))
    replacements = (
      ("“", ''), ("”", ''), ("//", ""), ("«", ''), ("»",''), (",", ''),
      (";", ''), (".", ''),
    #   ("?", ''), ("¿", ''), ("¡", ''), ("!", ''), ("-", ' '),
    )
    for replacement in replacements:
        output = output.replace(*replacement)
    # Any sequence of two or more spaces should be converted into one space
    output = re.sub(r'(?is)\s+', ' ', output)
    return output.strip()


def metric2binary(meter, pad=11):
    return ([1 if syllable == "+" else 0 for syllable in meter] + [0] * (11 - len(meter)))[:pad]


def label2metric(label):
    return "".join("+" if l else "-" for l in label)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

es_test = (pd
    .read_json(open("adso100.json"))
    .query("manually_checked == True")[["line_text", "metrical_pattern"]]
    .assign(
        line_text=lambda x: x["line_text"].apply(clean_text),
        length=lambda x: x["metrical_pattern"].str.len()
    )
    .drop_duplicates("line_text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter"})
)
es_test = es_test[es_test["length"] == 11]
pos_names = [f"pos{i}" for i in range(1, 12)]
pos_labels = es_test.meter.apply(metric2binary)
es_test["labels"] = pos_labels
es_test[pos_names] = pos_labels.tolist()
es_test[pos_names] = es_test[pos_names].astype(float)
es = (pd
    .read_json(open("adso.json"))
    .query("manually_checked == True")[["line_text", "metrical_pattern"]]
    .assign(
        line_text=lambda x: x["line_text"].apply(clean_text),
        length=lambda x: x["metrical_pattern"].str.len()
    )
    .drop_duplicates("line_text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter"})
)
es = es[~es["text"].isin(es_test["text"])][es["length"] == 11]
pos_labels = es.meter.apply(metric2binary)
es["labels"] = pos_labels
es[pos_names] = pos_labels.tolist()
es[pos_names] = es[pos_names].astype(float)

es_train, es_val = train_test_split(
    es[["text"] + pos_names], test_size=0.25, random_state=42
)

#vectors_filename = "glove-sbwc.i25.vec"
#vectors_filename = "SBW-vectors-300-min5.vec"
#vectors_filename = "fasttext-SUC-embeddings-l-model.vec"
vectors_filename = "cc.es.300.vec"

if True: # not os.path.isfile("embeddings_index.pkl"):
    def get_coefs(word, *vector):
        return word, np.asarray(vector, dtype='float32')
    with open(f"/home/jupyter/{vectors_filename}") as vec_file:
        embeddings_index = dict(
            get_coefs(*line.strip().split())
            for idx, line in enumerate(tqdm(vec_file))
            if idx != 0  # First line continas counts
        )
#    with open("embeddings_index.pkl", "wb") as embeddings_file:
#        dill.dump(embeddings_index, embeddings_file)
else:
    with open("embeddings_index.pkl", "rb") as embeddings_file:
        embeddings_index = dill.load(embeddings_file)

embed_size = 300  # how big is each word vector
max_features = 7500  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 24  # max number of words per input
y_train = es_train[pos_names].values
y_test = es_test[pos_names].values
y_val = es_val[pos_names].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(es_train.text.values))
tokenized_train = tokenizer.texts_to_sequences(es_train.text.values)
tokenized_val = tokenizer.texts_to_sequences(es_val.text.values)
tokenized_test = tokenizer.texts_to_sequences(es_test.text.values)
X_train = pad_sequences(tokenized_train, maxlen=maxlen, padding="post")
X_val = pad_sequences(tokenized_val, maxlen=maxlen, padding="post")
X_test = pad_sequences(tokenized_test, maxlen=maxlen, padding="post")

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        break
    embedding_vector = embeddings_index.get(word, embeddings_index.get(word.lower()))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

inputs = Input(shape=maxlen)
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inputs)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation=tfa.activations.gelu)(x)
#x = TimeDistributed(Dense(50, activation=tfa.activations.gelu))(x)  # a dense layer as suggested by neuralNer
x = Dropout(0.1)(x)
x = Dense(11, activation="sigmoid")(x)

model_2lstm = Model(inputs=inputs, outputs=x)
model_2lstm.compile(loss='binary_crossentropy', optimizer=tfa.optimizers.AdamW(weight_decay=1e-4), metrics=['accuracy'])

model_2lstm.summary()
tf.keras.utils.plot_model(model_2lstm, show_shapes=True)

history10_2lstm = model_2lstm.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1,
)  # validation_split=0.15
print(history10_2lstm)

print(model_2lstm.evaluate(X_test, y_test, verbose=1))

history100_2lstm = model_2lstm.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    verbose=1,
)  # validation_split=0.15
print(history100_2lstm)

print(model_2lstm.evaluate(X_test, y_test, verbose=1))
