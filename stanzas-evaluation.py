#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import json
import logging
import os
import re
import sklearn
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import wandb
#from IPython import get_ipython
from simpletransformers.classification import MultiClassClassificationModel
from sklearn.model_selection import train_test_split


truthy_values = ("true", "1", "y", "yes")
TAG = os.environ.get("TAG", "bertsification")
MODELNAME = os.environ.get("MODELNAME", "bert;bert-base-multilingual-cased")
OVERWRITE = os.environ.get("OVERWRITE", "False").lower() in truthy_values
logging.basicConfig(level=logging.INFO, filename=time.strftime("models/{}-%Y-%m-%dT%H%M%S.log".format(TAG)))
with open('pid', 'w') as pid:
    pid.write(str(os.getpid()))
logging.info("Experiment '{}' on {}, (eval_df = {}, pid = {})".format(
    TAG, MODELNAME, str(os.getpid()),
))

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


def prepare_data():
    df = (pd
        .read_csv("/shared/stanzas-evaluation.csv")
        .rename(columns={"Stanza_text": "text", "ST_Correct": "labels"})
        .assign(
            text=lambda x: x["text"].apply(clean_text),
            labels=lambda x: "unknown" if str(x) == "None" else x,
        )
    )
    train_df, eval_df = train_test_split(
        df, stratify=df["labels"], test_size=0.25, random_state=42
    )
    num_labels = len(df["labels"].unique())
    return train_df, eval_df, num_labels


def train_model(train_df, num_labels):
    model_type, model_name = MODELNAME.split(";")
    model_output = 'models/{}-{}-{}'.format(TAG, model_type, model_name.replace("/", "-"))
    if OVERWRITE is False and os.path.exists(model_output):
        logging.info("Skipping training of {}".format(model_name))
        sys.exit(0)
    logging.info("Starting training of {}".format(model_name))
    run = wandb.init(project=model_output.split("/")[-1], reinit=True)
    model = MultiClassClassificationModel(
        model_type, model_name, num_labels=11, args={
            'output_dir': model_output,
            'best_model_dir': '{}/best'.format(model_output),
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'use_cached_eval_features': True,
            'num_train_epochs': 3,  # For BERT, 2, 3, 4
            'save_steps': 1000,
            'evaluate_during_training': False,
            'evaluate_during_training_steps': 1000,
            'early_stopping_delta': 0.00001,
            'manual_seed': 42,
            # 'learning_rate': 2e-5,  # For BERT, 5e-5, 3e-5, 2e-5
            # For BERT 16, 32. It could be 128, but with gradient_acc_steps set to 2 is equivalent
            'train_batch_size': 8 if "large" in model_name else 32,
            'eval_batch_size': 8 if "large" in model_name else 32,
            # Doubles train_batch_size, but gradients and wrights are calculated once every 2 steps
            'gradient_accumulation_steps': 2 if "large" in model_name else 1,
            'max_seq_length': 64,
            'wandb_project': model_output.split("/")[-1],
            #'wandb_kwargs': {'reinit': True},
            # "adam_epsilon": 3e-5,  # 1e-8
            "silent": False,
            "fp16": False,
            "n_gpu": 1,
    })
    # train the model
    model.train_model(train_df)
    return model, run


def eval_model(model, eval_df, run):
    eval_df["predicted"], *_ = model.predict(eval_df.text.values)
    acc = sum(eval_df.labels == eval_df.predicted) / eval_df.labels.size
    logging.info("Accuracy: {}".format(acc))
    wandb.log({"accuracy_es": acc})
    run.finish()


def main() -> None:
    logging.info("Starting...")
    train, eval_df, num_labels = prepare_data()
    model, run = train_model(train, num_labels)
    eval_model(model, eval_df, run)
    logging.info("Done.")


if __name__ == "__main__":
    main()
