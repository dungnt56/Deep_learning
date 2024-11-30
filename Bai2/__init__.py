import numpy as np

# Hàm kích hoạt (Hàm bước)
def activation_function(u):
    return 1 if u >= 0 else -1

# Hàm dự đoán (Predict)
def predict(weights, bias, X):
    u = np.dot(X, weights) + bias  # u = W.X + B
    return activation_function(u)

# Hàm huấn luyện (Train)
def train_perceptron(X, y, learning_rate=0.01, epochs=100):
    n_features = X.shape[1]  # Số đặc trưng
    weights = np.zeros(n_features)  # Khởi tạo trọng số ban đầu
    bias = 0  # Bias ban đầu bằng 0

    # Huấn luyện qua nhiều epoch
    for epoch in range(epochs):
        for i in range(len(X)):
            # Dự đoán đầu ra
            prediction = predict(weights, bias, X[i])
            # Cập nhật trọng số nếu sai
            if prediction != y[i]:
                weights += learning_rate * (y[i] - prediction) * X[i]
                bias += learning_rate * (y[i] - prediction)

    return weights, bias

# Hàm chính để chạy chương trình
if __name__ == "__main__":
    # Nhập số lượng điểm m và số đặc trưng n
    m = int(input("Nhập số lượng dữ liệu (m): "))
    n = int(input("Nhập số lượng đặc trưng (n): "))

    # Sinh dữ liệu ngẫu nhiên (hoặc nhập tay)
    print("Sinh dữ liệu ngẫu nhiên...")
    X = np.random.randint(0, 10, (m, n))  # Ma trận m x n
    y = np.random.choice([-1, 1], m)  # Nhãn -1 hoặc 1

    print("Dữ liệu X (đầu vào):")
    print(X)
    print("Nhãn y (đầu ra):")
    print(y)

    # Huấn luyện Perceptron
    learning_rate = 0.01
    epochs = 100
    weights, bias = train_perceptron(X, y, learning_rate, epochs)

    print(f"Trọng số cuối cùng: {weights}, Bias: {bias}")

    # Dự đoán trên dữ liệu đã học
    print("Dự đoán trên dữ liệu đã học:")
    for i in range(len(X)):
        prediction = predict(weights, bias, X[i])
        print(f"Dữ liệu: {X[i]}, Nhãn thực tế: {y[i]}, Dự đoán: {prediction}")
