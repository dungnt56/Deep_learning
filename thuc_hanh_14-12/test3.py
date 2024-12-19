import numpy as np

# Define patterns (AND Gate in bipolar form)
patterns = np.array([
    [-1, -1, -1],
    [-1,  1, -1],
    [ 1, -1, -1],
    [ 1,  1,  1]
])

n_neurons = patterns.shape[1]

weights = np.zeros((n_neurons, n_neurons))
for pattern in patterns:
    weights += np.outer(pattern, pattern)

np.fill_diagonal(weights, 0)

print("Weight Matrix:")
print(weights)

def hopfield_recall(input_pattern, weights, max_iterations=10):
    state = np.copy(input_pattern)
    for _ in range(max_iterations):
        for i in range(len(state)):
            net_input = np.dot(weights[i], state)
            state[i] = 1 if net_input > 0 else -1
    return state

test_pattern = [1, 1, -1]
output = hopfield_recall(test_pattern, weights)

print("Input Pattern:", test_pattern)
print("Recalled Pattern:", output)
