import os
from typing import Tuple, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.debugger import verbose_debug_msg


class NetMLP(nn.Module):
    """
    PyTorch Multilayer perceptron.
    """

    def __init__(self, input_size, layer_sizes, activation=nn.ReLU(), epochs=10, lr=0.001, l2reg=0.0001, dropout=0.1) -> None:
        super(NetMLP, self).__init__()

        # Layer definition
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_sizes[0])])
        self.layers.extend(nn.Linear(layer_sizes[i-1], layer_sizes[i]) for i in range(1, len(layer_sizes)))
        self.layers.append(nn.Linear(layer_sizes[-1], 1))

        # Define activation
        self.activation = activation
        self.finalAct = nn.Sigmoid()

        # optimization
        self.epochs = epochs
        self.lr = lr
        self.l2reg = l2reg

        # Loss & optimization
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(lr=lr, weight_decay=l2reg, momentum=0.8, params=self.parameters())

    def forward(self, x) -> Any:
        """
        Forward pass in MLP
        :param x:
        :return:
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(self.dropout(x)))

        x = self.finalAct(self.layers[-1](x))

        return x

    def compute_loss(self, x, y) -> Tuple[float, float, float, float, float]:
        preds = self.forward(x)
        loss = self.criterion(preds, y)

        out = (preds > 0.5).float()
        acc, prec, rec, f1 = [metric(y.cpu(), out.cpu()) for metric in [accuracy_score, precision_score, recall_score, f1_score]]

        return loss, acc, prec, rec, f1

    def eval_loader(self, loader) -> Dict:
        metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        for i, (x, y) in enumerate(loader):
            loss, acc, prec, rec, f1 = self.compute_loss(x, y)
            metrics["loss"] += loss.item()
            metrics["accuracy"] += acc
            metrics["precision"] += prec
            metrics["recall"] += rec
            metrics["f1"] += f1

        for k in metrics.keys():
            metrics[k] /= len(loader)

        return metrics

    def fit(self, trainset, valset) -> Dict:

        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'val_accuracy': [], 'val_precision': [],
                   'recall': [], 'f1': [], 'val_recall': [], 'val_f1': []}

        for epoch in range(self.epochs):
            train_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

            for i, (x, y) in enumerate(trainset):
                self.optimizer.zero_grad()
                loss, acc, prec, rec, f1 = self.compute_loss(x, y)
                train_metrics["loss"] += loss.item()
                train_metrics["accuracy"] += acc
                train_metrics["precision"] += prec
                train_metrics["recall"] += rec
                train_metrics["f1"] += f1

                loss.backward()
                self.optimizer.step()

            for k in train_metrics.keys():
                train_metrics[k] /= len(trainset)
                history[k].append(train_metrics[k])

            value_metrics = self.eval_loader(valset)

            for k in value_metrics:
                history['val_' + k].append(value_metrics[k])

            if epoch % 5 == 0:
                verbose_debug_msg(
                    msg=f"Current epoch: {epoch}. Loss: {train_metrics['loss']}  -  "
                        f"Accuracy: {train_metrics['accuracy']}  -  Prec.: {train_metrics['precision']}  -  "
                        f"Rec.: {train_metrics['recall']}  -  F1: {train_metrics['f1']}", level=2
                )

        return history


class NetCNN(nn.Module):
    """
    Implementation of a text-processing CNN
    """

    def __init__(self, vocab_size, emb_matrix, filter_size=[1, 2, 3, 4, 5], num_filter=16, emb_size=200, emb_tune=False,
                 epochs=10, lr=0.001, l2reg=0.0001, dropout=0.1):
        super(NetCNN, self).__init__()

        # NN hyperparameters
        self.filter_sizes = filter_size
        self.num_filters = num_filter
        self.emb_size = emb_size

        # NN Layer design
        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = emb_tune
        self.cn1 = nn.ModuleList([nn.Conv2d(1, self.num_filters, (x, self.emb_size)) for x in self.filter_sizes])
        self.dropout = dropout
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, 1)
        self.finAct = nn.Sigmoid()

        # Optimization paramters
        self.epochs = epochs
        self.lr = lr
        self.l2reg = l2reg

        # Loss & optimization
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2reg)

    def forward(self, x: torch.tensor) -> torch.Tensor:

        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(cn(x)).squeeze(3) for cn in self.cn1]
        x = [F.max_pool1d(i, i.size(2).squeeze(2)) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)

        return self.finAct(self.fc(x))

    def compute_loss(self, x: torch.tensor, y: torch.tensor) -> Tuple[Any, Any, Any, Any, Any]:

        pred = self.forward(x)
        loss = self.criterion(pred, y)
        out = (pred > 0.5).float()

        acc, prec, rec, f1 = [metric(y.cpu(), out.cpu()) for metric in [accuracy_score, precision_score,
                                                                            recall_score, f1_score]]

        return loss, acc, prec, rec, f1

    def eval_data(self, data) -> Dict:
        metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        for i, (x, y) in enumerate(data):
            loss, acc, pre, rec, f1 = self.compute_loss(x, y)

            metrics["loss"] += loss.item()
            metrics["accuracy"] += acc
            metrics["precision"] += pre
            metrics["recall"] += rec
            metrics["f1"] += f1

        for k in metrics.keys():
            metrics[k] /= len(data)

        return metrics

    def fit(self, trainset, valset) -> Dict:

        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'val_accuracy': [], 'val_precision': [],
                   'recall': [], 'f1': [], 'val_recall': [], 'val_f1': []}

        for epoch in range(self.epochs):
            train_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

            for i, (x, y) in enumerate(trainset):
                self.optimizer.zero_grad()
                loss, acc, prec, rec, f1 = self.compute_loss(x, y)
                train_metrics["loss"] += loss.item()
                train_metrics["accuracy"] += acc
                train_metrics["precision"] += prec
                train_metrics["recall"] += rec
                train_metrics["f1"] += f1

                loss.backward()
                self.optimizer.step()

            for k in train_metrics.keys():
                train_metrics[k] /= len(trainset)
                history[k].append(train_metrics[k])

            value_metrics = self.eval_loader(valset)

            for k in value_metrics:
                history['val_' + k].append(value_metrics[k])

            if epoch % 5 == 0:
                verbose_debug_msg(
                    msg=f"Current epoch: {epoch}. Loss: {train_metrics['loss']}  -  "
                        f"Accuracy: {train_metrics['accuracy']}  -  Prec.: {train_metrics['precision']}  -  "
                        f"Rec.: {train_metrics['recall']}  -  F1: {train_metrics['f1']}", level=1
                )

        return history
