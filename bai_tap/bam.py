import numpy as np

input_size = 5
output_size = 4
weights = np.zeros((input_size, output_size))

input_patterns = [
    [1, -1, 1, -1, 1],
    [1, -1, 1, -1, -1],
    [-1, 1, -1, 1, 1]
]

output_patterns = [
    [1, 1, 1, 1],
    [-1, 1, 1, -1],
    [1, -1, -1, 1]
]

for x, y in zip(input_patterns, output_patterns):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(1, -1)
    weights += np.dot(x, y)

def recall(input_pattern, weights, steps=5):
    x = np.array(input_pattern)
    for _ in range(steps):
        y = np.sign(np.dot(x, weights))
        x = np.sign(np.dot(y, weights.T))
    return x, y

input_test = [1, -1, 1, -1, 1]
recalled_input, recalled_output = recall(input_test, weights)

print("Input test pattern:", input_test)
print("Recalled input pattern:", recalled_input)
print("Recalled output pattern:", recalled_output)
