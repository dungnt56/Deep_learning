def perceptron(inputs, weights, bias):
    """
    Hàm mô phỏng một nơron nhân tạo.
    :param inputs: Danh sách giá trị đầu vào [x1, x2, ...]
    :param weights: Danh sách trọng số tương ứng [w1, w2, ...]
    :param bias: Giá trị bias
    :return: Output của nơron (0 hoặc 1)
    """
    # Tính tổng trọng số: z = w1*x1 + w2*x2 + ... + bias
    z = sum(i * w for i, w in zip(inputs, weights)) + bias
    # Hàm kích hoạt Step (hàm ngưỡng)
    return 1 if z >= 0 else 0
