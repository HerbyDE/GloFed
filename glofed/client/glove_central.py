import torch.cuda
from data.preprocessor import PreProcessor
from data.data_handler import DataHandler
from models.glove_models import NetMLP, NetCNN
from utils.type_conversion import cast_to_float, cast_to_int
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn


import numpy as np
import pandas as pd

from typing import Dict, Tuple, Any


class GloVeLoader():

    def __init__(self):

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

    def transform_data(self, model_variant):

        # Insert TorchLoader
        if model_variant == "MLP":
            print("In MLP branch")
            self.X_train, self.Y_train = self.preprocessor.vectorize_data(self.trainset, self.w2vec)
            self.X_test, self.Y_test = self.preprocessor.vectorize_data(self.testset, self.w2vec)
            self.X_val, self.Y_val = self.preprocessor.vectorize_data(self.valset, self.w2vec)

            self.torch_train = TensorDataset(cast_to_float(self.X_train), cast_to_float(self.Y_train))
            self.torch_val = TensorDataset(cast_to_float(self.X_val), cast_to_float(self.Y_val))
            self.torch_test = TensorDataset(cast_to_float(self.X_test), cast_to_float(self.Y_test))

        elif model_variant == "CNN":
            print("In CNN branch")
            self.X_train, self.Y_train = self.preprocessor.tokenize_data(self.trainset, self.w2id)
            self.X_test, self.Y_test = self.preprocessor.tokenize_data(self.testset, self.w2id)
            self.X_val, self.Y_val = self.preprocessor.tokenize_data(self.valset, self.w2id)

            self.torch_train = TensorDataset(cast_to_int(self.X_train), cast_to_float(self.Y_train))
            self.torch_val = TensorDataset(cast_to_int(self.X_val), cast_to_float(self.Y_val))
            self.torch_test = TensorDataset(cast_to_int(self.X_test), cast_to_float(self.Y_test))

        else:
            raise KeyError("Please make sure to specify the correct model. Options are MLP and CNN.")

        return

    def build_emb_matrix(self, trainset) -> Tuple[Dict, np.array]:
        vocab = self.preprocessor.generate_vocab(trainset)
        w2id, emb_mx = self.preprocessor.embed_vocab(voc=vocab, w2v=self.w2vec)

        return w2id, emb_mx

    def mlp_model(self, epochs=50):

        train = DataLoader(self.torch_train, shuffle=True, batch_size=self.batch_size)
        val = DataLoader(self.torch_val, shuffle=True, batch_size=self.batch_size)
        test = DataLoader(self.torch_test, shuffle=True, batch_size=self.batch_size)

        mlp = NetMLP(input_size=200, layer_sizes=[128, 32], activation=nn.Tanh(), epochs=epochs, lr=0.0002,
                     l2reg=0.00005, dropout=0)

        mlp = mlp.cuda() if torch.cuda.is_available() else mlp
        print("Number of MLP parameters: {}".format(sum([np.prod(p.size()) for p in mlp.parameters()])))

        print("Starting MLP training...")
        history = mlp.fit(train, val)

        print("Evaluating model performance...")
        perform = mlp.eval_loader(test)

        print("MLP performance: ", perform)
        print("MLP Training & Testing done!")

    def cnn_model(self):
        train = DataLoader(self.torch_train, shuffle=True, batch_size=self.batch_size)
        val = DataLoader(self.torch_val, shuffle=True, batch_size=self.batch_size)
        test = DataLoader(self.torch_test, shuffle=True, batch_size=self.batch_size)
        emb = self.build_emb_matrix(trainset=self.trainset)

        cnn = NetCNN(vocab_size=emb[1].shape[0], emb_matrix=emb[1], filter_size=[1, 2, 3, 5, 10], num_filter=8,
                     emb_size=emb[1].shape[1], emb_tune=False, epochs=150, lr=0.000002, l2reg=0.0003, dropout=0.1)

        cnn = cnn.cuda() if torch.cuda.is_available() else cnn

        print(cnn)
        print(sum([np.prod(p.size()) for p in cnn.parameters()]) - np.prod(self.build_emb_matrix(trainset=train)[1].shape))

        history = cnn.fit(train, val)
        test_performance = cnn.eval_data(test)
        print("Test performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}".format(
            *[test_performance[m] for m in ['loss', 'accuracy', 'precision', 'recall', 'f1']]))


