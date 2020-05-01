import numpy as np

# two layer neurons, the first one feeds the second

inputs = [[0.5, 2, 1, -0.8],
          [0.5, 2, 1, -0.8],
          [0.5, 2, 1, -0.8]]

weights1 = [[0.8, 1, -1, 0.5],
           [0.2, 0.9, 1, -0.2],
           [2, 1.5, -1, -0.5]]

biases1 = [1, -1, 2]

layer_outputs1 = np.dot(inputs, np.transpose(weights1)) + biases1


weights2 = [[1, 2, -1],
           [0.9, 0.1, 1.4],
           [0.88, 0.5, -1]]

biases2 = [0.8, -1, 3]

layer_outputs2 = np.dot(layer_outputs1, np.transpose(weights2)) + biases2

print(layer_outputs2)
