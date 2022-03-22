import os
from client.glofed_client import FederatedMLPClient


# if __name__ == "__main__":
#
#     # Configure environment variables
#     os.environ["DEBUG_VERBOSITY"] = "1"
#
#     # Set seed
#     torch.manual_seed(1234)
#     np.random.seed(1234)
#
#     gc = GloVeLoader()
#     gc.load_datasets(trainset="training.1600000.processed.noemoticon.csv", testset="testdata.manual.2009.06.14.csv")
#     gc.load_embedding()
#     gc.transform_data()
#     gc.mlp_model()
#     # gc.cnn_model()

if __name__ == "__main__":

    os.environ["DEBUG_VERBOSITY"] = "1"

    FederatedMLPClient().launch()
