#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import argparse
import json
import logging
import os
import re
import time
import string
import sys
from itertools import product
from pathlib import Path

import fasttext as ft
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pattern2indexed(pattern: str) -> list:
    return np.where(np.array(list(pattern)) == "+")[0].tolist()

def prepare_data(lang, data_dir):
    data_path = Path(data_dir)
    if lang == "es":
        return prepare_data_es(data_path)
    elif lang == "de":
        return prepare_data_de(data_path)
    elif lang == "en":
        return prepare_data_en(data_path)


def prepare_data_es(data_path) -> None:
    es_test = (pd
        .read_json(open(data_path / "adso100.json"))
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
        .read_json(open(data_path / "adso.json"))
        .query("manually_checked == True")[["line_text", "metrical_pattern"]]
        .assign(
            line_text=lambda x: x["line_text"].apply(clean_text),
            length=lambda x: x["metrical_pattern"].str.len()
        )
        .drop_duplicates("line_text")
        .rename(columns={"line_text": "text", "metrical_pattern": "meter"})
    )
    es = es[~es["text"].isin(es_test["text"])][es["length"] == 11]
    with open(data_path / "adso100.txt", "w") as file:
        for entry in es_test.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    with open(data_path / "adso.txt", "w") as file:
        for entry in es.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    return data_path / "adso.txt", data_path / "adso100.txt"


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
    with test.open() as test_file:
        total = 0
        matches = 0
        for line in tqdm(test_file):
            text = re.split("__label__syll-\d+", line)[-1].strip()
            labels = set(line.replace(text, "").split())
            predicted = set(model.predict(text, k=-1, threshold=.5)[0])
            if labels == predicted:
                matches += 1
            total += 1
    accuracy = matches / total
    print(f"Accuracy: {accuracy}")


def prepare_data_en(data_path):
    en_test = (pd
        .read_csv(data_path / "4b4v_prosodic_meter.csv")
        .assign(
            text=lambda x: x["text"].apply(clean_text),
            length=lambda x: x["meter"].str.len()
        )
        .drop_duplicates("text")
        .rename(columns={"line_text": "text", "metrical_pattern": "meter", "prosodic_meter": "sota"})
    )
    en_test = en_test.query("length in (5,6,7,8,9,10,11)")
    # if not os.path.isfile("ecpa.json"):
    #     get_ipython().system("averell export ecpa --filename ecpa.json")
    en = (pd
        .read_json(open(data_path / "ecpa.json"))
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
    en = en[~en["text"].isin(en_test["text"])].query("length in (5,6,7,8,9,10,11)")
    with open(data_path / "4b4v.txt", "w") as file:
        for entry in en_test.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    with open(data_path / "ecpa.txt", "w") as file:
        for entry in en.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    return data_path / "ecpa.txt", data_path / "4b4v.txt"


def prepare_data_de(data_path):
    ge = (pd
        .read_csv(data_path / "po-emo-metricalizer.csv")
        .rename(columns={"verse": "text", "annotated_pattern": "meter", "metricalizer_pattern": "sota"})
        .assign(
            text=lambda x: x["text"].apply(clean_text),
            length=lambda x: x["meter"].str.len()
        )
        .drop_duplicates("text")
        .query("length in (5, 6, 7, 8, 9, 10, 11)")
    )
    ge_train, ge_test = train_test_split(ge, test_size=0.15, random_state=42)
    with open(data_path / "poemo.txt", "w") as file:
        for entry in ge_train.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    with open(data_path / "poemo_test.txt", "w") as file:
        for entry in ge_test.to_dict(orient="records"):
            text = entry["text"]
            indices = pattern2indexed(entry["meter"])
            labels = [f"__label__syll-{index + 1}" for index in indices]
            file.write(f"{' '.join(labels)} {text}\n")
    return data_path / "poemo.txt", data_path / "poemo_test.txt"


def main(args: argparse.ArgumentParser) -> None:
    print(args)
    train, test = prepare_data(args.lang, args.data)
    model = ft.train_supervised(
        input=str(train),
        pretrainedVectors=args.vectors,  # cc.es.300.vec, wiki.es.align.vec
        dim=args.dims,
        lr=args.lr,
        epoch=args.epochs,
        wordNgrams=args.ngrams,
        bucket=args.bucket,
        loss=args.loss,
        thread=args.threads,
    )
    model.save_model(args.output)
    test_results = model.test(str(test), k=-1, threshold=.5)
    print("Samples: {}, Precision: {}, Recall: {}".format(*test_results))
    eval_model(model, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", type=str)
    parser.add_argument("vectors", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("dims", type=int, default=300)
    parser.add_argument("lr", type=float, default=0.25)
    parser.add_argument("epochs", type=int, default=10)
    parser.add_argument("loss", type=str, default="ova")
    parser.add_argument("--data", metavar="data", type=str, default="data")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ngrams", type=int, default=3)
    parser.add_argument("--bucket", type=int, default=2000000)
    args = parser.parse_args()
    main(args)
