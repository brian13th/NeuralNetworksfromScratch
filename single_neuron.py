import numpy as np

# single neurons with three inputs

inputs = [1, 0.8, -2]

weights = [0.5, -1, 1]

bias = 2

outputs = np.dot(weights, inputs) + bias

print(round(outputs,1))
