

class InputData:

    net_data = None  # type: []
    loss_data = None  # type: []

    def __init__(self, net_data, loss_data):
        self.net_data = net_data
        self.loss_data = loss_data
