import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# just doing sigmoid activation functions because it's easier
class FeedForwardNeuralNetwork:
    def __init__(self, hidden_width, depth, input_size, output_size):
        self.width = hidden_width
        self.depth = depth
        self.network = []

        # activations are in columns
        self.network.append(np.random.normal(0, 1, (input_size + 1, hidden_width)))
        for i in range(depth):
            self.network.append(np.random.normal(0, 1, (self.network[-1].shape[1] + 1, hidden_width)))
        self.network.append(np.random.normal(0, 1, (self.network[-1].shape[1] + 1, output_size)))

    def feed_forward(self, dataset):

        ff_data = dataset.transpose()

        # +2 to account for input and output weights
        for i in range(self.depth+2):
            ff_data = np.concatenate((np.ones((1, ff_data.shape[1])), ff_data))
            ff_data = sigmoid(np.matmul(self.network[i].transpose(), ff_data))

        return ff_data

