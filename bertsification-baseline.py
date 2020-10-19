#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import json
import logging
import os
import re
import time
import string
import sys
from itertools import product

import fasttext as ft
import numpy as np
import pandas as pd
# import wandb
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pattern2indexed(pattern: str) -> list:
    return np.where(np.array(list(pattern)) == "+")[0].tolist()


def prepare_data() -> None:
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
    with open("data/adso100.txt", "w") as file:
        for entry in es_test.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    with open("data/adso.txt", "w") as file:
        for entry in es.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")


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


def eval_model(model, test):
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

    es_test["pred"] = es_test.text.apply(lambda x: sorted([
        int(label.replace("__label__syll-", "").strip()) - 1
        for label in model.predict(x, k=-1, threshold=.5)[0]
    ]))
    es_test["y"] = es_test.meter.apply(pattern2indexed)
    print((es_test.y == es_test.pred).sum() / es_test.shape[0])


def main() -> None:
    prepare_data()
    model = ft.train_supervised(
        input="data/adso.txt",
        pretrainedVectors="cc.es.300.vec",  # "wiki.es.align.vec",
        dim=300,
        lr=0.25,
        epoch=10,
        wordNgrams=3,
        bucket=2000000,
        loss='ova',
        thread=1,
    )
    model.save_model("bertsification-baseline-es.bin")
    print(model.test("data/adso100.txt", k=-1))
    eval_model(model, "data/adso100.txt")


if __name__ == "__main__":
    main()
