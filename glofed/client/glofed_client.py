"""
The federated client inherits the centralized model.
"""
import flwr as fl

from typing import List, Dict, Tuple
from collections import OrderedDict

import numpy as np
import socket
import torch
from torch.utils.data import DataLoader

from models.glove_models import NetMLP
from client.glove_central import GloVeLoader
from utils.debugger import verbose_debug_msg


class FederatedMLPClient(fl.client.NumPyClient):

    def __init__(self):
        super(FederatedMLPClient, self).__init__()
        self.loader = GloVeLoader()
        self.batch_size = 128

        self.loader.load_datasets(trainset="training.1600000.processed.noemoticon.csv",
                                  testset="testdata.manual.2009.06.14.csv")
        self.loader.load_embedding()
        self.loader.transform_data()

        self.trainset = DataLoader(self.loader.torch_train, shuffle=True, batch_size=self.batch_size)
        self.valset = DataLoader(self.loader.torch_val, shuffle=True, batch_size=self.batch_size)
        self.testset = DataLoader(self.loader.torch_test, shuffle=True, batch_size=self.batch_size)

        self.df_len = {
            "trainset": len(self.trainset.dataset),
            "valset": len(self.valset.dataset),
            "testset": len(self.testset.dataset)
        }

        verbose_debug_msg("Instatiating MLP model.")
        self.model = NetMLP(input_size=200, layer_sizes=[128, 32], activation=torch.nn.Tanh(), epochs=50, lr=0.0002,
                            l2reg=0.00005, dropout=0)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, params: List[np.ndarray]) -> None:
        params = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, params: List[np.ndarray], cfg: Dict[str, str]):
        self.set_parameters(params)
        self.model.fit(self.trainset, self.valset)
        return self.get_parameters(), self.df_len["trainset"], {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        performance = self.model.eval_loader(loader=self.testset)

        verbose_debug_msg(f"Model performance: {performance}", level=1)
        return performance["loss"], self.df_len["testset"], {}

    def launch(self) -> None:
        verbose_debug_msg(f"Initializing FL client", level=1)
        fl.client.start_numpy_client("0.0.0.0:8080", self)





