import numpy as np

def step_function(z):
    return 1 if z > 0 else 0

def train_perceptron(samples, learning_rate):
    num_features = samples.shape[1] - 1  # Số lượng đầu vào (trừ cột đầu ra)
    weights = np.random.rand(num_features + 1)  # Khởi tạo trọng số ngẫu nhiên (+1 cho bias)
    epoch = 0

    while True:
        total_error = 0
        epoch += 1
        for sample in samples:
            X = np.array([-1] + list(sample[:num_features]))  # X0 = -1 (bias), X1, X2, X3...
            d = sample[num_features]  # Đầu ra mong muốn
            U = np.dot(weights, X)  # Tính đầu ra thực tế
            y = step_function(U)
            e = d - y
            total_error += abs(e)
            weights += learning_rate * e * X  # Cập nhật trọng số
        if total_error == 0:
            break
    return weights

# Mẫu dữ liệu cho 3 đầu vào (A, B, C, d)
samples_and = np.array([
    [0, 0, 0, 0],  # A, B, C, d
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],  # AND logic
])

# Huấn luyện perceptron
final_weights = train_perceptron(samples_and, learning_rate=0.15)

# Kiểm tra kết quả trên tất cả các mẫu
print("\nKiểm tra kết quả trên các mẫu:")
for sample in samples_and:
    X = np.array([-1] + list(sample[:3]))  # X0 = -1 (bias), X1, X2, X3
    y_a = step_function(np.dot(final_weights, X))
    print(f"Input: {sample[:3]}, Predicted: {y_a}, Desired: {sample[3]}")
