import numpy as np

np.random.seed(0)

# input data
X = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):  # n_inputs are the features of the inputs (columns in a mxn matrix)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)    # multiply with 0.1 to restrain weight values range to -0.1 to 0.1
        self.biases = np.zeros((1, n_neurons))  # biases are a matrix of zeros for now. Usually we use bias no move a "dead" neuron

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)