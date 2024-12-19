import numpy as np

# Hàm kích hoạt sgn
def sgn(x):
    return np.where(x >= 0, 1, -1)

# Định nghĩa tập dữ liệu AND Gate
inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
outputs = np.array([-1, -1, -1, 1])  # Đầu ra AND Gate

# Khởi tạo trọng số (Hebb rule)
n_neurons = inputs.shape[1]  # Số nơron
weights = np.zeros((n_neurons, n_neurons))

# Cập nhật trọng số dựa trên đầu vào và đầu ra
for x, y in zip(inputs, outputs):
    weights += np.outer(x, x)  # Quy tắc Hebb

# Đặt trọng số tự kết nối (diagonal) bằng 0
np.fill_diagonal(weights, 0)

print("Trọng số của mạng Hopfield:")
print(weights)

# Kiểm tra mạng với từng mẫu đầu vào
print("\nKết quả cập nhật mạng:")
for x in inputs:
    print(f"Đầu vào: {x}")
    state = x
    for _ in range(10):  # Cập nhật tối đa 10 lần để hội tụ
        net = np.dot(weights, state)
        new_state = sgn(net)
        if np.array_equal(new_state, state):  # Nếu trạng thái không đổi, dừng lại
            break
        state = new_state
    print(f"Trạng thái đầu ra: {state}")
