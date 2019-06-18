import numpy as np


def sigmoid(x, derivative=False):
    """Return the sigmoid (or derivative of sigmoid, if chosen) of x."""
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


# just doing sigmoid activation functions because it's easier
class FeedForwardNeuralNetwork:
    """Contains the necessary information and algorithms to train a simple logistic feedforward neural network"""

    def __init__(self, hidden_width, depth, input_size, output_size):
        """Initialize the class and create the network.

        Args:
            hidden_width (int): the width of the hidden layers of the network.
            depth (int): the number of hidden layers in the network.
            input_size (int): the number of features in the input dataset.
            output_size (int): the number of outputs.

        Creates the network as a list of numpy arrays.
        Weights are assigned from a Gaussian distribution with mean 0 and standard deviation 0.
        """
        self.width = hidden_width
        self.depth = depth
        self.network = []

        # activations are in columns
        # minimum one hidden layer
        # depth = 0 => perceptron
        if depth == 0:
            self.network.append((np.random.normal(0, 1, (input_size + 1, output_size))))
            return

        # deep NN with normally distributed initial weights
        self.network.append(np.random.normal(0, 1, (input_size + 1, hidden_width)))
        for i in range(depth - 1):
            self.network.append(np.random.normal(0, 1, (self.network[-1].shape[1] + 1, hidden_width)))
        self.network.append(np.random.normal(0, 1, (self.network[-1].shape[1] + 1, output_size)))

        return

    def feed_forward(self, dataset, save_activations=False):

        ff_data = dataset.transpose()
        activations = [ff_data]

        # +2 to account for input and output weights
        for i in range(self.depth+2):
            ff_data = np.concatenate((np.ones((1, ff_data.shape[1])), ff_data))
            ff_data = sigmoid(self.network[i].transpose() @ ff_data)
            if save_activations:
                activations.append(ff_data)

        if save_activations:
            return activations
        else:
            return ff_data

    def back_prop(self, activations, dataset_output, alpha, weight_decay=0):
        # TODO: test the backpropagation algorithm
        m = activations[0].shape[0]
        g = (dataset_output - activations[-1]) / (activations[-1] * (1 - activations[-1])) / m

        for k in range(self.depth, 0, -1):
            # calculate gradient
            g = alpha * g * sigmoid(self.network[k].transpose() @ activations[k], derivative=True) \
                               * activations[k-1]
            self.network[k] += g

            # add weight decay to gradient calculation
            if weight_decay != 0: # don't change the bias terms, so we replace the top row of weights with zeros
                self.network[k] += weight_decay / m \
                                   * np.concatenate(np.zeros(1, self.network[k].shape[1]), self.network[k][1:, :])

            g *= self.network[k]


    def train(self, dataset, epochs):
        # TODO: implement training
        pass


    def check_gradient(self, dataset):
        # TODO: implement gradient checking by comparing with numerical derivative
        pass

    def cost(self, dataset_input, dataset_output, weight_decay=0):
        h = sigmoid(self.feed_forward(dataset_input))

        # logistic cost function
        j = np.sum(-dataset_output * np.log(h) - (1 - dataset_output) * np.log(1 - h)) / dataset_input.shape[0]
        j_reg = weight_decay / 2 * sum([np.sum(np.square(arr)) for arr in self.network]) / dataset_input.shape(0)

        return j + j_reg
