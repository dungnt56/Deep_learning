import numpy as np


# Hàm step function
def step_function(z):
    return 1 if z > 0 else 0


# Huấn luyện perceptron
def train_perceptron(samples, learning_rate):
    # Khởi tạo trọng số ban đầu
    weights = np.array([0.1, 0.2, 0.3])  # W0 (bias), W1, W2
    print(f"Trọng số ban đầu: {weights}")

    count_loop = 0  # Đếm số vòng lặp
    while True:  # Vòng lặp chạy vô hạn, chỉ dừng khi tổng sai số = 0
        total_error = 0
        count_loop += 1
        print(f"\nEpoch {count_loop}:")

        for sample in samples:
            # Chuẩn bị đầu vào
            X = np.array([-1, sample[0], sample[1]])  # X0 = -1 (bias), A, B
            d = sample[2]  # Đầu ra mong muốn

            # Tính đầu ra thực tế
            U = np.dot(weights, X)
            y = step_function(U)

            # Tính sai số
            e = d - y
            total_error += abs(e)

            # Cập nhật trọng số nếu có sai số
            weights += learning_rate * e * X

            # Hiển thị thông tin chi tiết
            print(f"Mẫu: {sample[:2]}, d: {d}, y: {y}, Sai số: {e}, Trọng số: {weights}")

        # Dừng nếu tổng sai số = 0
        if total_error == 0:
            print("Huấn luyện hoàn tất!")
            break

    return weights


# Samples AND: (A, B, dy_d)
samples_and = np.array([
    [0, 0, 0],  # A, B, d
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
])

# Thực thi huấn luyện với η = 0.15
final_weights = train_perceptron(samples_and, 0.15)

# Hiển thị kết quả trọng số cuối cùng
print("\nTrọng số cuối cùng:", final_weights)

# Kiểm tra kết quả trên tất cả các mẫu
print("\nKiểm tra kết quả trên các mẫu:")
for sample in samples_and:
    X = np.array([-1, sample[0], sample[1]])  # X0 = -1 (bias), A, B
    y_a = step_function(np.dot(final_weights, X))
    print(f"Mẫu: {sample[:2]}, Dự đoán: {y_a}, Mong muốn: {sample[2]}")
