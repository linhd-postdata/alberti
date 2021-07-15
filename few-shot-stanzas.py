#!/usr/bin/env python3
import io
import os
import zipfile

import pandas as pd
import torch
from torch.utils.data import Dataset

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import argparse
import random

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

import learn2learn as l2l


# truthy_values = ("true", "1", "y", "yes")
# TAG = os.environ.get("TAG", "bertsification")
# MODELNAME = os.environ.get("MODELNAME", "bert;bert-base-multilingual-cased")
# OVERWRITE = os.environ.get("OVERWRITE", "False").lower() in truthy_values
# logging.basicConfig(level=logging.INFO, filename=time.strftime("models/{}-%Y-%m-%dT%H%M%S.log".format(TAG)))
# with open('pid', 'w') as pid:
#     pid.write(str(os.getpid()))
# logging.info("Experiment '{}', (eval_df = {}, pid = {})".format(
#     TAG, MODELNAME, str(os.getpid()),
# ))

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


def clean_labels(label):
    return "unknown" if str(label) == "None" else label


class StanzasDataset(Dataset):

    def __init__(self, datafile, train=True, transform=None):
        self.transform = transform
        if transform == 'roberta':
            # self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large', force_reload=True)
            self.roberta = torch.hub.load('pytorch/fairseq', 'xlmr.large', force_reload=True)
            # self.roberta.eval()  # disable dropout (or leave in train mode to finetune)
        self.df = (pd
            .read_csv(datafile)
            .rename(columns={"Stanza_text": "text", "ST_Correct": "stanza"})
            .assign(
                text=lambda x: x["text"].apply(clean_text),
                stanza=lambda x: x["stanza"].apply(clean_labels),
            )
        )
        self.label_encoder = LabelEncoder()
        self.df["labels"] = self.label_encoder.fit_transform(self.df["stanza"])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            return (
                self.roberta.encode(self.df['text'][idx]),
                self.df['labels'][idx]
            )
        return self.df['text'][idx], self.df['labels'][idx]


class FewShotNet(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_dim=768, inner_dim=200, pooler_dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.output = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.log_softmax(self.output(x), dim=1)
        return x


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class _BatchedDataset(torch.utils.data.Dataset):
    def __init__(self, batched):
        self.sents = [s for s in batched[0]]
        self.ys = [y for y in batched[1]]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return (self.sents[idx], self.ys[idx])


def compute_loss(task, roberta, device, learner, loss_func, batch=15):
    loss = 0.0
    acc = 0.0
    for i, (x, y) in enumerate(torch.utils.data.DataLoader(
            _BatchedDataset(task), batch_size=batch, shuffle=True, num_workers=0)):
        # RoBERTa ENCODING
        x = collate_tokens([roberta.encode(sent) for sent in x], pad_idx=1)
        with torch.no_grad():
            x = roberta.extract_features(x)
        x = x[:, 0, :]

        # Moving to device
        x, y = x.to(device), y.view(-1).to(device)

        output = learner(x)
        curr_loss = loss_func(output, y)
        acc += accuracy(output, y)
        loss += curr_loss / len(task)
    loss /= len(task)
    return loss, acc


def main(lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, device=torch.device("cpu"),
         datafile="/tmp/text"):
    dataset = StanzasDataset(datafile=datafile)
    dataset = l2l.data.MetaDataset(dataset)

    classes = list(range(len(dataset.labels))) # 45 classes
    random.shuffle(classes)

    train_dataset, validation_dataset, test_dataset = dataset, dataset, dataset

    train_gen = l2l.data.TaskDataset(
            train_dataset, num_tasks=20000,
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(
                    train_dataset, n=ways, k=shots, filter_labels=classes[:20]),
                l2l.data.transforms.LoadData(train_dataset),
                l2l.data.transforms.RemapLabels(train_dataset)],)

    validation_gen = l2l.data.TaskDataset(
            validation_dataset, num_tasks=20000,
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(
                    validation_dataset, n=ways, k=shots, filter_labels=classes[20:30]),
                l2l.data.transforms.LoadData(validation_dataset),
                l2l.data.transforms.RemapLabels(validation_dataset)],)

    test_gen = l2l.data.TaskDataset(
            test_dataset, num_tasks=20000,
            task_transforms=[
                l2l.data.transforms.FusedNWaysKShots(
                    test_dataset, n=ways, k=shots, filter_labels=classes[30:]),
                l2l.data.transforms.LoadData(test_dataset),
                l2l.data.transforms.RemapLabels(test_dataset)],)

    # torch.hub.set_dir(datafile)
    roberta = torch.hub.load('pytorch/fairseq', 'xlmr.large', force_reload=True)
    roberta.eval()
    roberta.to(device)
    model = FewShotNet(num_classes=ways)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction="sum")

    tqdm_bar = tqdm(range(iterations))

    accs = []
    for iteration in tqdm_bar:
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task, valid_task = train_gen.sample(), validation_gen.sample()

            # Fast Adaptation
            for step in range(fas):
                train_error, _ = compute_loss(train_task, roberta, device, learner, loss_func, batch=shots * ways)
                learner.adapt(train_error)

            # Compute validation loss
            valid_error, valid_acc = compute_loss(valid_task, roberta, device, learner, loss_func,
                                                  batch=shots * ways)
            iteration_error += valid_error
            iteration_acc += valid_acc

        iteration_error /= tps
        iteration_acc /= tps
        tqdm_bar.set_description("Loss : {:.3f} Acc : {:.3f}".format(iteration_error.item(), iteration_acc))
        accs.append(iteration_acc)
        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()
    print (f'first and best validation accuracy: {accs[0]:.4f}, {max(accs):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn Text Classification Example')

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=1, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=32, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--datafile', type=str, default="./stanzas-evaluation.csv", metavar='S',
                        help='Path to CSV with stanza evaluation data')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    main(lr=args.lr, maml_lr=args.maml_lr, iterations=args.iterations, ways=args.ways, shots=args.shots,
         tps=args.tasks_per_step, fas=args.fast_adaption_steps, device=device,
         datafile=args.datafile)
