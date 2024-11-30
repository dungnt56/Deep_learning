import numpy as np

def step_function(z):
    return 1 if z > 0 else 0

def train_perceptron(samples, learning_rate):
    weights = np.array([0.1, 0.2, 0.3])
    epoch = 0
    while True:
        total_error = 0
        epoch += 1
        for sample in samples:
            X = np.array([-1, sample[0], sample[1]])
            d = sample[2]
            U = np.dot(weights, X)
            y = step_function(U)
            e = d - y
            total_error += abs(e)
            weights += learning_rate * e * X
        if total_error == 0:
            break
    return weights

samples_and = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
])

final_weights = train_perceptron(samples_and, learning_rate=0.15)

for sample in samples_and:
    X = np.array([-1, sample[0], sample[1]])
    y_a = step_function(np.dot(final_weights, X))
    print(f"Input: {sample[:2]}, Predicted: {y_a}, Desired: {sample[2]}")
