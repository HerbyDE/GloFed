import torch.cuda
from data.preprocessor import PreProcessor
from data.data_handler import DataHandler
from models.glove_models import NetMLP, NetCNN
from utils.type_conversion import cast_to_float, cast_to_int
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn


import numpy as np
import pandas as pd


class CentralGloVe():

    def __init__(self, trainset="training.1600000.processed.noemoticon.csv", testset="testdata.manual.2009.06.14.csv", embedding="glove.twitter.27B.200d.txt"):

        # Datasets
        self.data_loader = DataHandler()
        self.trainset_loc = ""
        self.testset_loc = ""
        self.trainset = pd.DataFrame()
        self.testset = pd.DataFrame()
        self.valset = pd.DataFrame()

        # Embeddings
        self.emb_loc = ""
        self.w2vec = {}
        self.w2id = {}

        # Data vectors
        self.X_train = self.X_test = self.X_val = self.Y_train = self.Y_test = self.Y_val = pd.DataFrame()

        # Pre-processor
        self.preprocessor = PreProcessor()

        # Parameters
        self.batch_size = 128

    def load_trainset(self, trainset="training.1600000.processed.noemoticon.csv"):
        self.data_files = self.data_loader.download_dataset()
        self.trainset_loc = self.data_files["directory"] + trainset
        return self.trainset_loc

    def load_testset(self, testset: str="testdata.manual.2009.06.14.csv"):
        self.data_files = self.data_loader.download_dataset()
        self.testset_loc = self.data_files["directory"] + testset
        return self.testset_loc

    def load_datasets(self, trainset: str, testset: str) -> None:

        print("Downloading training dataset...")
        trainset = self.load_trainset(trainset)

        print("Downloading test dataset...")
        testset = self.load_testset(testset)

        print("Generating dataframes...")
        self.trainset, self.valset, self.testset = self.preprocessor.load_and_prep_datasets(trainset=trainset,
                                                                                            testset=testset,
                                                                                            n_train=25000 , n_val=8000)
        return None

    def load_embedding(self, embedding_name="glove.twitter.27B.200d.txt"):

        print("Downloading word embedding...")
        emb_files = self.data_loader.download_embedding()
        self.emb_loc = emb_files["directory"] + embedding_name
        self.w2vec, self.w2id = self.preprocessor.load_embedding(self.emb_loc)

        return self.w2vec, self.w2id

    def transform_data(self):

        self.X_train, self.Y_train = self.preprocessor.vectorize_data(self.trainset, self.w2vec)
        self.X_test, self.Y_test = self.preprocessor.vectorize_data(self.testset, self.w2vec)
        self.X_val, self.Y_val = self.preprocessor.vectorize_data(self.valset, self.w2vec)

        # Insert TorchLoader
        self.torch_train = TensorDataset(cast_to_float(self.X_train), cast_to_float(self.Y_train))
        self.torch_val = TensorDataset(cast_to_float(self.X_val), cast_to_float(self.Y_val))
        self.torch_test = TensorDataset(cast_to_float(self.X_test), cast_to_float(self.Y_test))
        return

    def mlp_model(self):

        train = DataLoader(self.torch_train, shuffle=True, batch_size=self.batch_size)
        val = DataLoader(self.torch_val, shuffle=True, batch_size=self.batch_size)
        test = DataLoader(self.torch_test, shuffle=True, batch_size=self.batch_size)

        mlp = NetMLP(input_size=200, layer_sizes=[128, 32], activation=nn.Tanh(), epochs=50, lr=0.0002,
                     l2reg=0.00005, dropout=0)

        mlp = mlp.cuda() if torch.cuda.is_available() else mlp
        print("Number of MLP parameters: {}".format(sum([np.prod(p.size()) for p in mlp.parameters()])))

        print("Starting MLP training...")
        history = mlp.fit(train, val)

        print("Evaluating model performance...")
        perform = mlp.eval_loader(test)

        print("MLP performance: ", perform)
        print("MLP Training & Testing done!")
