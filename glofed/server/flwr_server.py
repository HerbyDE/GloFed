import flwr as fl


class FLServer(object):

    def __init__(self):
        pass

    def start(self, ip, port, rounds=3):
        host = f"{ip}{f':{port}' if port else ''}"

        fl.server.start_server(server_address=host, config={"rounds": rounds})


if __name__ == "__main__":
    FLServer().start(ip="0.0.0.0", port=8080, rounds=3)
