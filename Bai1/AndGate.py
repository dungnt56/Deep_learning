from Perceptron import perceptron
# Trọng số và bias cho cổng OR
weights_or = [1, 1]
bias_or = -0.5

# Các giá trị đầu vào cho cổng OR
inputs_or = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

print("\nCổng OR:")
for inp in inputs_or:
    output = perceptron(inp, weights_or, bias_or)
    print(f"Đầu vào: {inp}, Đầu ra: {output}")
