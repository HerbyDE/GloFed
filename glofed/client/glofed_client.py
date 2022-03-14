"""
The federated client inherits the centralized model.
"""
import flwr as fl


class FederatedMLPClient(fl.client.NumPyClient):

    def __init__(self):
        super(FederatedMLPClient, self).__init__()
