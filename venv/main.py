import numpy as np
X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0]
])

y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
np.random.seed(1)
# Тегла за входния към скрития слой
weights_input_hidden = 2 * np.random.random((3, 4)) - 1  # 3 входа -> 4 неврона в скрит слой
# Тегла за скрития към изходния слой
weights_hidden_output = 2 * np.random.random((4, 3)) - 1  # 4 неврона -> 3 изхода
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
for epoch in range(10000):  # Брой итерации
    # Forward propagation
    input_layer = X
    hidden_layer = sigmoid(np.dot(input_layer, weights_input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output))
    
    output_error = y - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
    
    weights_hidden_output += np.dot(hidden_layer.T, output_delta)
    weights_input_hidden += np.dot(input_layer.T, hidden_delta)

test_input = np.array([1, 0, 0])
hidden_layer = sigmoid(np.dot(test_input, weights_input_hidden))
test_output = sigmoid(np.dot(hidden_layer, weights_hidden_output))

print("Предсказание за новите данни:", test_output)
