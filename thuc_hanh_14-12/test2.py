import numpy as np

def sgn(x):
    return np.where(x > 0, 1, -1)

patterns = [
    [-1, -1, -1],
    [-1,  1, -1],
    [ 1, -1, -1],
    [ 1,  1,  1]
]

n_neurons = 3

W = np.zeros((n_neurons, n_neurons))
for p in patterns:
    p = np.array(p)
    W += np.outer(p, p)

# Loại bỏ trọng số tự kết nối (w_ii = 0)
np.fill_diagonal(W, 0)

print("Ma trận trọng số W:")
print(W)

state = np.array([-1, 1, -1])

for _ in range(15):
    state = sgn(np.dot(W, state))
    print("Trạng thái cập nhật:", state)
