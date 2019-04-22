import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


# just doing sigmoid activation functions because it's easier
class FeedForwardNeuralNetwork:
    def __init__(self, hidden_width, depth, input_size, output_size):
        self.width = hidden_width
        self.depth = depth
        self.network = []

        # activations are in columns
        # minimum one hidden layer
        self.network.append(np.random.normal(0, 1, (input_size + 1, hidden_width)))
        for i in range(depth):
            self.network.append(np.random.normal(0, 1, (self.network[-1].shape[1] + 1, hidden_width)))
        self.network.append(np.random.normal(0, 1, (self.network[-1].shape[1] + 1, output_size)))

    def feed_forward(self, dataset):

        ff_data = dataset.transpose()

        # +2 to account for input and output weights
        for i in range(self.depth+2):
            ff_data = np.concatenate((np.ones((1, ff_data.shape[1])), ff_data))
            ff_data = sigmoid(self.network[i].transpose() @ ff_data)

        return ff_data

    def train(self, dataset, epochs):
        # TODO: implement training
        pass

    def _backprop(self, gradient):
        # TODO: implement backprop
        pass

    def check_gradient(self, dataset):
        # TODO: implement gradient checking by comparing with numerical derivative
        pass

    def _cost(self, dataset_input, dataset_output, weight_decay):
        h = sigmoid(self.feed_forward(dataset_input) - dataset_output)

        # logistic cost function
        J = np.sum(-dataset_output * np.log(h) - (1 - dataset_output) * np.log(1 - h)) / dataset_input.shape[0]
        J_reg = weight_decay / 2 * sum([np.sum(np.square(arr)) for arr in self.network]) / dataset_input.shape(0)

        return J + J_reg


