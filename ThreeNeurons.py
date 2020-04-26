import numpy as np

# three neurons are feeded by four neuron outputs (from hidden layer to hidden layer case)

inputs = [0.5, 2, 1, -0.8]

weights = [[0.8, 1, -1, 0.5],
           [0.2, 0.9, 1, -0.2],
           [2, 1.5, -1, -0.5]]

biases = [1, -1, 2] #three biases as we have three neuros and each one has it's own output gravity

outputs = np.dot(weights, inputs) + biases
print(outputs)