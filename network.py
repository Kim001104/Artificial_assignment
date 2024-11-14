import csv
import os
import numpy as np  

# 하이퍼파라미터
hidden_layers = [2000,500]
learning_rate = 0.001
epochs = 10
input_size = 4096
output_size = 111

# 가중치 및 편향 초기화
weights = [np.random.randn(input_size, hidden_layers[0])]
biases = [np.zeros(hidden_layers[0])]

# 히든 레이어 간의 가중치 및 편향 추가
for i in range(1, len(hidden_layers)):
    weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
    biases.append(np.zeros(hidden_layers[i]))

# 출력층 가중치 및 편향 추가
weights.append(np.random.randn(hidden_layers[-1], output_size))
biases.append(np.zeros(output_size))

# 데이터 로드 함수
def load_data(sub_dir, file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '한글글자데이터', sub_dir, file_name)
    data = []
    labels = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])
            row_data = [float(value) for value in row[1:]]
            data.append(row_data)
            labels.append(label)
    return np.array(data), np.array(labels)

# 데이터 로드
train_data, train_labels = load_data('train', 'train_data.csv')

# 원-핫 인코딩 함수
def one_hot_encoding(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels

train_labels = one_hot_encoding(train_labels, output_size)

# 시그모이드 함수(순전파 사용)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수 미분(역전파 사용)
def sigmoid_derivative(x):
    return x * (1 - x)

# 소프트 맥스(출력층 활성화 함수로 사용)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 순전파 함수
def forward_propagation(input_data, weights, biases):
    activations = [input_data]
    for i in range(len(weights)-1):
        input_data = sigmoid(np.dot(input_data, weights[i]) + biases[i])
        activations.append(input_data)
    input_data = softmax(np.dot(input_data, weights[-1]) + biases[-1])
    activations.append(input_data)
    return activations

# 교차 엔트로피 손실 함수(softmax와 함께 사용)
def cross_entropy_loss(output, target):
    return -np.sum(target * np.log(output))

# 역전파 함수
def backpropagation(activations, target, weights):
    deltas = [activations[-1] - target]  # 출력층 오차
    gradients_w = []
    gradients_b = []
    
    for i in reversed(range(len(weights))):
        layer_activation = activations[i]
        delta = deltas[-1]
        grad_w = np.dot(layer_activation.reshape(-1, 1), delta.reshape(1, -1))
        grad_b = delta

        gradients_w.insert(0, grad_w)
        gradients_b.insert(0, grad_b)

        if i > 0:
            delta = np.dot(delta, weights[i].T) * sigmoid_derivative(activations[i])
            deltas.append(delta)

    return gradients_w, gradients_b

# 가중치 및 편향 업데이트 함수
def update_weights(weights, biases, gradients_w, gradients_b, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients_w[i]
        biases[i] -= learning_rate * gradients_b[i]

# 학습 과정
for epoch in range(epochs):
    
    total_error = 0
    correct_predictions = 0
    for i in range(len(train_data)):
        input = train_data[i]
        target = train_labels[i]

        # 순전파
        activations = forward_propagation(input, weights, biases)
        output = activations[-1]

        # # MSE를 통한 오차 계산
        # error = np.mean((output - target) ** 2)
        # total_error += error

        #교차엔트로피 함수를 통한 오차계산
        loss = cross_entropy_loss(output, target)
        total_error += loss

        # 정확도 계산
        if np.argmax(output) == np.argmax(target):
            correct_predictions += 1

        # 역전파 및 가중치 업데이트
        gradients_w, gradients_b = backpropagation(activations, target, weights)
        update_weights(weights, biases, gradients_w, gradients_b, learning_rate)

    # 평균 오차 및 정확도 출력
    average_error = total_error / len(train_data)
    accuracy = correct_predictions / len(train_data) * 100
    print(f"Epoch {epoch+1}/{epochs}, 평균 오차: {average_error:.4f}, 정확도: {accuracy:.2f}%")

