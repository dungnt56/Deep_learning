import numpy as np


def step_function(z):
    return 1 if z > 0 else 0


def train_perceptron(samples, learning_rate):
    num_features = samples.shape[1] - 1
    weights = np.random.rand(num_features + 1)
    epoch = 0

    while True:
        total_error = 0
        epoch += 1
        for sample in samples:
            X = np.array([-1] + list(sample[:num_features]))  # Thêm bias
            d = sample[num_features]
            U = np.dot(weights, X)
            y = step_function(U)
            e = d - y
            total_error += abs(e)
            weights += learning_rate * e * X
        if total_error == 0:
            break
    return weights


def create_samples():
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],  # AND logic
    ])


def evaluate_perceptron(weights, samples):
    print("\nKiểm tra kết quả trên các mẫu:")
    for sample in samples:
        X = np.array([-1] + list(sample[:4]))  # X0 = -1 (bias), X1, X2, X3, X4
        y_a = step_function(np.dot(weights, X))
        print(f"Input: {sample[:4]}, Predicted: {y_a}, Desired: {sample[4]}")

samples_and = create_samples()
final_weights = train_perceptron(samples_and, learning_rate=0.15)
evaluate_perceptron(final_weights, samples_and)
