#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import argparse
import json
import logging
import os
import re
import sklearn
import time

import numpy as np
import pandas as pd
import wandb
#from IPython import get_ipython
from simpletransformers.classification import MultiLabelClassificationModel


# logging.basicConfig(level=logging.INFO, filename=time.strftime("bertsification-%Y-%m-%dT%H%M%S.log"))
logging.basicConfig(level=logging.INFO)
with open('pid', 'w') as pid:
    pid.write(str(os.getpid()))
logging.info("Starting with pid = {}".format(str(os.getpid())))


# Utils
def metric2binary(meter, pad=12):
    return ([1 if syllable == "+" else 0 for syllable in meter] + [0] * (pad - len(meter)))[:pad]


def label2metric(label):
    return "".join("+" if l else "-" for l in label)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def compute_accuracy(model, df):
    df["predicted"], *_ = model.predict(df.text.values)
    df["predicted"] = df["predicted"].apply(label2metric)
    df["pred"] = df.apply(lambda x: str(x.predicted)[:int(x.length)], axis=1)
    return sum(df.meter == df.pred) / df.meter.size


def evaluate(model, model_name, model_lang, data_lang, eval_df, test_df, sota):
    eval_acc = compute_accuracy(model, eval_df)
    test_acc = compute_accuracy(model, test_df)
    logging.info("Accuracy EVAL [{}:{}]: {} ({})".format(model_lang, data_lang, eval_acc, model_name))
    logging.info("Accuracy TEST [{}:{}]: {} ({})".format(model_lang, data_lang, test_acc, model_name))
    logging.info("Accuracy SOTA [{}:{}]: {}".format(model_lang, data_lang, sota))
    wandb.log({
        "eval:accuracy": eval_acc,
        "test:accuracy": test_acc,
        "{}/eval:accuracy".format(data_lang): eval_acc,
        "{}/test:accuracy".format(data_lang): test_acc,
        "{}/sota".format(data_lang): sota,
    })


BASE_URL = "https://storage.googleapis.com/postdata-models/bertsification/data"

def main(args):
    assert args.lang in ("es", "en", "de", "multi")
    # Spanish
    es_train = pd.read_csv(f"{BASE_URL}/es2_train.csv")
    es_eval = pd.read_csv(f"{BASE_URL}/es2_dev.csv")
    es_test = pd.read_csv(f"{BASE_URL}/es2_test.csv")
    es_train["labels"] = es_train.meter.apply(metric2binary)
    es_eval["labels"] = es_eval.meter.apply(metric2binary)
    es_test["labels"] = es_test.meter.apply(metric2binary)
    # es_sota = 0.9623  # From Rantanplan
    es_sota = sum(es_test.meter == es_test.sota) / es_test.meter.size

    # English
    en_train = pd.read_csv(f"{BASE_URL}/en_train.csv")
    en_eval = pd.read_csv(f"{BASE_URL}/en_dev.csv")
    en_test = pd.read_csv(f"{BASE_URL}/en_test.csv")
    en_train["labels"] = en_train.meter.apply(metric2binary)
    en_eval["labels"] = en_eval.meter.apply(metric2binary)
    en_test["labels"] = en_test.meter.apply(metric2binary)
    en_sota = sum(en_test.meter == en_test.sota) / en_test.meter.size

    # German
    de_train = pd.read_csv(f"{BASE_URL}/de_train.csv")
    de_eval = pd.read_csv(f"{BASE_URL}/de_dev.csv")
    de_test = pd.read_csv(f"{BASE_URL}/de_test.csv")
    de_train["labels"] = de_train.meter.apply(metric2binary)
    de_eval["labels"] = de_eval.meter.apply(metric2binary)
    de_test["labels"] = de_test.meter.apply(metric2binary)
    de_sota = sum(de_test.meter == de_test.sota) / de_test.meter.size

    # Training
    # Multilingual inputs
    # - bert bert-base-multilingual-cased
    # - distilbert distilbert-base-multilingual-cased
    # - xlmroberta, xlm-roberta-base
    # - xlmroberta, xlm-roberta-large
    # Only English
    # - roberta roberta-base
    # - roberta roberta-large
    # - albert albert-xxlarge-v2

    # You can set class weights by using the optional weight argument
    # models = (
    # #    ("xlnet", "xlnet-base-cased"),
    #     ("bert", "bert-base-multilingual-cased"),
    #     ("distilbert", "distilbert-base-multilingual-cased"),
    #     ("roberta", "roberta-base"),
    #     ("roberta", "roberta-large"),
    #     ("xlmroberta", "xlm-roberta-base"),
    #     ("xlmroberta", "xlm-roberta-large"),
    #     ("electra", "google/electra-base-discriminator"),
    # #    ("albert", "albert-base-v2"),
    # #    ("albert", "albert-xxlarge-v2"),
    # )
    model_type, model_name = args.model_name.split(":")
    model_output = 'models/{}-{}-{}'.format(args.lang, model_type, model_name.replace("/", "-"))
    logging.info("Starting training of {} for {}".format(model_name, args.lang))
    run_name = model_output.split("/", 1)[-1]
    training_args = {
        'output_dir': args.output_dir,
        'cache_dir': args.cache_dir,
        'best_model_dir': '{}/best'.format(model_output),
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'use_cached_eval_features': True,
        'num_train_epochs': args.num_train_epochs,  # For BERT, 2, 3, 4
        # 'save_steps': 500,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'no_save': True,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 500,
        'use_early_stopping': False,
        # 'early_stopping_metric': "accuracy_score",
        # 'early_stopping_patience': 10,
        # 'early_stopping_delta': 0.00001,
        'learning_rate': args.learning_rate,  # For BERT, 5e-5, 3e-5, 2e-5
        # For BERT 16, 32. It could be 128, but with gradient_acc_steps set to 2 is equivalent
        'train_batch_size': args.train_batch_size,  #  if "large" in model_name else 32,
        'eval_batch_size': args.eval_batch_size,  # if "large" in model_name else 32,
        # Doubles train_batch_size, but gradients and writes are calculated once every 2 steps
        'gradient_accumulation_steps': 1,  # 2 if "large" in model_name else 1,
        'max_seq_length': args.max_seq_length,
        'wandb_kwargs': {'name': run_name},  # 'reinit': True
        # "adam_epsilon": 3e-5,  # 1e-8
        "silent": False,
        "fp16": True,
        "n_gpu": 1,
        'manual_seed': args.seed,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
    }
    wandb.init(name=run_name, config=training_args)
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        num_labels=args.num_labels,
        args=training_args,
    )

    # Train the model
    if args.lang == "multi":
        train_df = pd.concat([es_train, en_train, de_train], ignore_index=True)
        eval_df = pd.concat([es_eval, en_eval, de_eval], ignore_index=True)
    elif args.lang == "es":
        train_df = es_train
        eval_df = es_eval
    elif args.lang == "en":
        train_df = en_train
        eval_df = en_eval
    elif args.lang == "de":
        train_df = de_train
        eval_df = de_eval
    logging.info("Training with")
    logging.info("\n" + str(train_df[["text", "labels"]].head(3)))
    logging.info("Evaluating with")
    logging.info("\n" + str(eval_df[["text", "labels"]].head(3)))
    model.train_model(
        train_df[["text", "labels"]],
        eval_df=eval_df[["text", "labels"]],
    )

    # Evaluate
    result, model_outputs, wrong_predictions = model.eval_model(
        eval_df[["text", "labels"]]
    )
    logging.info(str(result))

    # Test
    if args.lang in ("es", "multi"):
        evaluate(model, model_name, args.lang, "es", es_eval, es_test, es_sota)
    if args.lang in ("en", "multi"):
        evaluate(model, model_name, args.lang, "en", en_eval, en_test, en_sota)
    if args.lang in ("de", "multi"):
        evaluate(model, model_name, args.lang, "de", de_eval, de_test, de_sota)
    logging.info("Done training '{}'".format(model_output))

if __name__ == "__main__":
    # yesno = lambda x: str(x).lower() in {'true', 't', '1', 'yes', 'y'}
    parser = argparse.ArgumentParser(description="""
    Evaluating BERTsification finetuning
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_name',
        metavar='model_name', help='Model name or path: "bert:bert-base-multilingual-cased"'
    )
    parser.add_argument('--lang',
        metavar='lang', help='Language: es, en, de, multi'
    )
    parser.add_argument('--num_train_epochs',
        metavar='num_train_epochs', default=4, type=float,
        help='Number of training epochs',
    )
    parser.add_argument('--max_seq_length',
        metavar='max_seq_length', default=24, type=int,
        help='Max sequence length',
    )
    parser.add_argument('--num_labels',
        metavar='num_labels', default=12, type=int,
        help='Number of labels (syllables)',
    )
    parser.add_argument('--cache_dir',
        metavar='cache_dir', default="./cache/",
        help='Cache dir for the transformer library',
    )
    parser.add_argument('--output_dir',
        metavar='output_dir', default="./output/",
        help='Output dir for models and logs',
    )
    parser.add_argument('--overwrite_output_dir',
        metavar='overwrite_output_dir', default=True, type=bool,
        help='Overwrite output dir if present',
    )
    parser.add_argument('--seed',
        metavar='seed', type=int, default=2021,
        help='Seed for the experiments',
    )
    parser.add_argument('--train_batch_size',
        metavar='train_batch_size', type=int, default=8,
        help='Batch size for training',
    )
    parser.add_argument('--eval_batch_size',
        metavar='eval_batch_size', type=int,
        help='Batch size for evaluation. Defaults to train_batch_size',
    )
    parser.add_argument('--learning_rate',
        metavar='learning_rate', type=float, default="3e-05",
        help='Learning rate',
    )
    parser.add_argument('--warmup_ratio',
        metavar='warmup_ratio', type=float, default=0.0,
        help='Warmup steps as percentage of the total number of steps',
    )
    parser.add_argument('--weight_decay',
        metavar='weight_decay', type=float, default=0.0,
        help='Weight decay',
    )

    args = parser.parse_args()
    main(args)
