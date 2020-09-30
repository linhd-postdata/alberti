#!/usr/bin/env python
# coding: utf-8
# conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch
# wandb login XXX
import json
import logging
import os
import re
import time
from itertools import product

from IPython import get_ipython
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.model_selection import train_test_split

PREFIX = os.environ.get("PREFIX", "bertsification")
LANGS = [lang.strip() for lang in os.environ.get("LANGS", "es,ge,en,multi").lower().split(",")]
MODELNAME = os.environ.get("MODELNAME")
EVAL = os.environ.get("EVAL", "True").lower() in ("true", "1", "y", "yes")
logging.basicConfig(level=logging.INFO, filename=time.strftime("models/{}-%Y-%m-%dT%H%M%S.log".format(PREFIX)))
with open('pid', 'w') as pid:
    pid.write(str(os.getpid()))
logging.info("Experiment '{}' on {}, (eval = {}, pid = {})".format(
    PREFIX, LANGS, str(EVAL), str(os.getpid()),
))

# SimpleTransformers (based on HuggingFace/Transformers) for Multilingual Scansion
# We will be using `simpletransformers`, a wrapper of `huggingface/transformers` to fine-tune different BERT-based and other architecture models with support for Spanish.

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


def metric2binary(meter, pad=11):
    return ([1 if syllable == "+" else 0 for syllable in meter] + [0] * (11 - len(meter)))[:pad]


def label2metric(label):
    return "".join("+" if l else "-" for l in label)


# Spanish
if not os.path.isfile("adso100.json"):
    get_ipython().system("averell export adso100 --filename adso100.json")
if not os.path.isfile("adso.json"):
    get_ipython().system("averell export adso --filename adso.json")

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
es["labels"] = es.meter.apply(metric2binary)

es_train, es_eval = train_test_split(
    es[["text", "labels"]], test_size=0.25, random_state=42)
logging.info("Spanish")
logging.info("- Lines: {} train, {} eval, {} test".format(es_train.shape[0], es_eval.shape[0], es_test.shape[0]))

# English
en_test = (pd
    .read_csv("4b4v_prosodic_meter.csv")
    .assign(
        text=lambda x: x["text"].apply(clean_text),
        length=lambda x: x["meter"].str.len()
    )
    .drop_duplicates("text")
    .rename(columns={"line_text": "text", "metrical_pattern": "meter", "prosodic_meter": "sota"})
)
en_test = en_test.query("length in (5,6,7,8,9,10,11)")
if not os.path.isfile("ecpa.json"):
    get_ipython().system("averell export ecpa --filename ecpa.json")
en = (pd
    .read_json(open("ecpa.json"))
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
en["labels"] = en.meter.apply(metric2binary)
en_train, en_eval = train_test_split(
    en[["text", "labels"]], test_size=0.25, random_state=42)
logging.info("English")
logging.info("- Lines: {} train, {} eval, {} test".format(en_train.shape[0], en_eval.shape[0], en_test.shape[0]))
# sota
en_sota = sum(en_test.meter == en_test.sota) / en_test.meter.size

# German
ge = (pd
    .read_csv("po-emo-metricalizer.csv")
    .rename(columns={"verse": "text", "annotated_pattern": "meter", "metricalizer_pattern": "sota"})
    .assign(
        text=lambda x: x["text"].apply(clean_text),
        length=lambda x: x["meter"].str.len()
    )
    .drop_duplicates("text")
    .query("length in (5, 6, 7, 8, 9, 10, 11)")
)
ge["labels"] = ge.meter.apply(metric2binary)

ge_train_eval, ge_test = train_test_split(ge, test_size=0.15, random_state=42)
ge_train, ge_eval = train_test_split(
    ge_train_eval[["text", "labels"]], test_size=0.176, random_state=42)
logging.info("German")
logging.info("- Lines: {} train, {} eval, {} test".format(ge_train.shape[0], ge_eval.shape[0], ge_test.shape[0]))
# sota
ge_sota = sum(ge_test.meter == ge_test.sota) / ge_test.meter.size

# training
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
models = (
#    ("xlnet", "xlnet-base-cased"),

    ("bert", "bert-base-multilingual-cased"),
    ("distilbert", "distilbert-base-multilingual-cased"),
    ("roberta", "roberta-base"),
    ("roberta", "roberta-large"),
    ("xlmroberta", "xlm-roberta-base"),
    ("xlmroberta", "xlm-roberta-large"),
    ("electra", "google/electra-base-discriminator"),

#    ("albert", "albert-base-v2"),
#    ("albert", "albert-xxlarge-v2"),
)
if MODELNAME:
    models = [MODELNAME.split(",")]
langs = LANGS or ("es", "ge", "en", "multi")
for lang, (model_type, model_name) in product(langs, models):
    logging.info("Starting training of {} for {}".format(model_name, lang))
    model_output = 'models/{}-{}-{}-{}'.format(PREFIX, lang, model_type, model_name.replace("/", "-"))
    model = MultiLabelClassificationModel(
        model_type, model_name, num_labels=11, args={
            'output_dir': model_output,
            'best_model_dir': '{}/best'.format(model_output),
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'use_cached_eval_features': True,
            'num_train_epochs': 100,
            'save_steps': 10000,
            'early_stopping_patience': 5,
            'evaluate_during_training': True,
            #'evaluate_during_training_steps': 1000,
            'manual_seed': 42,
            # 'learning_rate': 2e-5,
            'train_batch_size': 32,  # could be 128, but with gradient_acc_steps set to 2 is equivalent
            'eval_batch_size': 32,
            'gradient_accumulation_steps': 2,  # doubles train_batch_size, but gradients and wrights are calculated once every 2 steps
            'max_seq_length': 64,
            'use_early_stopping': True,
            'wandb_project': model_output.split("/")[-1],
            # "adam_epsilon": 3e-5,
            "silent": False,
            "fp16": False,
            "n_gpu": 1,
    })
    # train the model
    if lang == "multi":
        train_df = pd.concat([es_train, en_train, ge_train], ignore_index=True)
        eval_df = pd.concat([es_eval, en_eval, ge_eval], ignore_index=True)
    elif lang == "es":
        train_df = es_train
        eval_df = es_eval
    elif lang == "en":
        train_df = en_train
        eval_df = en_eval
    elif lang == "ge":
        train_df = ge_train
        eval_df = ge_eval
    model.train_model(train_df, eval_df=eval_df)
    # evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    logging.info(str(result))
    #logging.info(str(model_outputs))

    if lang in ("es", "multi"):
        es_test["predicted"], *_ = model.predict(es_test.text.values)
        es_test["predicted"] = es_test["predicted"].apply(label2metric)
        es_test["pred"] = es_test.apply(lambda x: str(x.predicted)[:int(x.length)], axis=1)
        es_bert = sum(es_test.meter == es_test.pred) / es_test.meter.size
        logging.info("Accuracy [{}:es]: {} ({})".format(lang, es_bert, model_name))
    if lang in ("en", "multi"):
        en_test["predicted"], *_ = model.predict(en_test.text.values)
        en_test["predicted"] = en_test["predicted"].apply(label2metric)
        en_test["pred"] = en_test.apply(lambda x: str(x.predicted)[:int(x.length)], axis=1)
        en_bert = sum(en_test.meter == en_test.pred) / en_test.meter.size
        logging.info("Accuracy [{}:en]: {} ({})".format(lang, en_bert, model_name))
    if lang in ("ge", "multi"):
        ge_test["predicted"], *_ = model.predict(ge_test.text.values)
        ge_test["predicted"] = ge_test["predicted"].apply(label2metric)
        ge_test["pred"] = ge_test.apply(lambda x: str(x.predicted)[:int(x.length)], axis=1)
        ge_bert = sum(ge_test.meter == ge_test.pred) / ge_test.meter.size
        logging.info("Accuracy [{}:ge]: {} ({})".format(lang, ge_bert, model_name))
    if lang in ("multi", ):
        test_df = pd.concat([es_test, en_test, ge_test], ignore_index=True)
        test_df["predicted"], *_ = model.predict(test_df.text.values)
        test_df["predicted"] = test_df["predicted"].apply(label2metric)
        test_df["pred"] = test_df.apply(lambda x: str(x.predicted)[:int(x.length)], axis=1)
        multi_bert = sum(test_df.meter == test_df.pred) / test_df.meter.size
        logging.info("Accuracy [{}:multi]: {} ({})".format(lang, multi_bert, model_name))
    logging.info("Done training '{}'".format(model_output))
    get_ipython().system("rm -rf `ls -dt models/{}-*/checkpoint*/ | awk 'NR>5'`".format(PREFIX))
logging.info("Done training")
