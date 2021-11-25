#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import json
import logging
import os
import re
import sklearn
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)

# Utils
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


def metric2binary(meter, pad=12):
    return (
        [1 if syllable == "+" else 0 for syllable in meter]
        + [0] * (pad - len(meter))
    )[:pad]


def label2metric(label):
    return "".join("+" if l else "-" for l in label)


# Spanish
logging.info("Spanish")
es_test = (pd
    .read_json(open("data/adso100.json"))
    .query("manually_checked == True")[["line_text", "metrical_pattern"]]
    .assign(
        line_text=lambda x: x["line_text"].apply(clean_text),
        length=lambda x: x["metrical_pattern"].str.len()
    )
    .drop_duplicates("line_text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter"})
)
es_test = es_test[es_test["length"] == 11]
es = (pd
    .read_json(open("data/adso.json"))
    .query("manually_checked == True")[["line_text", "metrical_pattern"]]
    .assign(
        line_text=lambda x: x["line_text"].apply(clean_text),
        length=lambda x: x["metrical_pattern"].str.len()
    )
    .drop_duplicates("line_text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter"})
)
es = es[~es["text"].isin(es_test["text"])][es["length"] == 11]
# es["labels"] = es.meter.apply(metric2binary)

es_train, es_dev = train_test_split(
    es, test_size=0.25, random_state=42)
es_sota = 0.9623  # From Rantanplan
logging.info("- Lines: {} train, {} eval, {} test".format(es_train.shape[0], es_dev.shape[0], es_test.shape[0]))
logging.info("- SOTA: {}".format(es_sota))
es_train.to_csv("data/es_train.csv", index=None)
es_dev.to_csv("data/es_dev.csv", index=None)
es_test.to_csv("data/es_test.csv", index=None)

# Spanish es2
logging.info("Spanish")
es = (pd
    .read_csv("data/adso_rantanplan.csv")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter", "rantanplan": "sota"})
    .assign(
        text=lambda x: x["text"].apply(clean_text),
        length=lambda x: x["meter"].str.len()
    )
    .drop_duplicates("text")
    .query("length in (11, )")
)
es_train_eval, es_test = train_test_split(es, test_size=0.15, random_state=42)
es_train, es_dev = train_test_split(
    es_train_eval.drop("sota", axis="columns"), test_size=0.176, random_state=42)
es_sota = sum(es_test.meter == es_test.sota) / es_test.meter.size
logging.info("- Lines: {} train, {} eval, {} test".format(es_train.shape[0], es_dev.shape[0], es_test.shape[0]))
logging.info("- SOTA: {}".format(es_sota))
es_train.to_csv("data/es2_train.csv", index=None)
es_dev.to_csv("data/es2_dev.csv", index=None)
es_test.to_csv("data/es2_test.csv", index=None)


# English
logging.info("English")
en_test = (pd
    .read_csv("data/4b4v_prosodic_meter.csv")
    .assign(
        text=lambda x: x["text"].apply(clean_text),
        length=lambda x: x["meter"].str.len()
    )
    .drop_duplicates("text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter", "prosodic_meter": "sota"})
)
en_test = en_test.query("length in (4,5,6,7,8,9,10,11,12)")
en = (pd
    .read_json(open("data/ecpa.json"))
    .query("manually_checked == True")[["line_text", "metrical_pattern"]]
    .assign(
        line_text=lambda x: x["line_text"].apply(clean_text),
        metrical_pattern=lambda x: x["metrical_pattern"].str.replace("|", "").str.replace("(", "").str.replace(")", "")
    )
    .assign(
        length=lambda x: x["metrical_pattern"].str.len(),
    )
    .drop_duplicates("line_text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter", "prosodic_meter": "sota"})
)
en = en[~en["text"].isin(en_test["text"])]
en = en.query("length in (4,5,6,7,8,9,10,11,12)")
en = en.query("meter != 'irregular'")
en = pd.concat([
    en[en.meter==m].sample(min(300, len(en[en.meter==m])))
    for m in en.meter.unique()
], axis=0)
en_total = en.meter.size + en_test.meter.size
en_train_size = int(en_total * .70)
en_dev_size = en_test_size = (en_total - en_train_size) // 2
en_test_sample_index = en_test.sample(
    en_test_size + en_train_size - en.meter.size
).index.tolist()
en = pd.concat([
    en, en_test[en_test.index.isin(en_test_sample_index)].drop("sota", axis="columns")
])
en_test = en_test[~en_test.index.isin(en_test_sample_index)]
en_train, en_dev = train_test_split(
    en, test_size=en_dev_size, stratify=en["length"], random_state=42)
# en_train = en_train[["text", "labels"]]
# en_dev = en_dev[["text", "labels"]]
en_sota = sum(en_test.meter == en_test.sota) / en_test.meter.size
logging.info("- Lines: {} train, {} eval, {} test".format(en_train.shape[0], en_dev.shape[0], en_test.shape[0]))
logging.info("- SOTA: {}".format(en_sota))
en_train.to_csv("data/en_train.csv", index=None)
en_dev.to_csv("data/en_dev.csv", index=None)
en_test.to_csv("data/en_test.csv", index=None)
# en["labels"] = en.meter.apply(metric2binary)

# German
logging.info("German")
ge = (pd
    .read_csv("data/po-emo-metricalizer.csv")
    .rename(columns={"verse": "text", "annotated_pattern": "meter", "metricalizer_pattern": "sota"})
    .assign(
        text=lambda x: x["text"].apply(clean_text),
        length=lambda x: x["meter"].str.len()
    )
    .drop_duplicates("text")
    .query("length in (4,5,6,7,8,9,10,11,12)")
)
# ge["labels"] = ge.meter.apply(metric2binary)

ge_train_eval, ge_test = train_test_split(ge, test_size=0.15, random_state=42)
ge_train, ge_dev = train_test_split(
    ge_train_eval.drop("sota", axis="columns"), test_size=0.176, random_state=42)
ge_sota = sum(ge_test.meter == ge_test.sota) / ge_test.meter.size
logging.info("- Lines: {} train, {} eval, {} test".format(ge_train.shape[0], ge_dev.shape[0], ge_test.shape[0]))
logging.info("- SOTA: {}".format(ge_sota))
ge_train.to_csv("data/ge_train.csv", index=None)
ge_dev.to_csv("data/ge_dev.csv", index=None)
ge_test.to_csv("data/ge_test.csv", index=None)


"""
Spanish (es2)
- Lines: 7095 train, 1516 eval, 1520 test
- SOTA: 0.9289473684210526
Spanish
- Lines: 6558 train, 2187 eval, 1401 test
- SOTA: 0.9623
English (en2)
- Lines: 1893 train, 406 eval, 406 test
- SOTA: 0.42857142857142855
German
- Lines: 774 train, 166 eval, 167 test
- SOTA: 0.4491017964071856
"""
