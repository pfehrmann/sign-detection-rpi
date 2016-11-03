from abc import ABCMeta, abstractmethod


class NeuralNet(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def train(self, list_of_images):
        pass

    @abstractmethod
    def test(self, list_of_images):
        pass

    @abstractmethod
    def validate(self, list_of_images):
        pass

    @abstractmethod
    def analyze(self, image):
        pass

    @abstractmethod
    def import_net(self, path_to_net):
        pass

    @abstractmethod
    def export_net(self, path_to_net):
        pass
