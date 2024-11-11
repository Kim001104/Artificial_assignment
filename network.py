import csv
import random
import math
import os
import numpy as np  

total_layers_num = 7       # 총 8층의 레이어들이 있음(인,아웃풋 포함)
epoch = 10          
lr = 0.1            # 학습률 0.1

image_size = 64     # bmp 파일의 1차원의 사이즈가 64 픽셀

input_node_num = image_size * image_size     # bmp 파일을 1차원 리스트로 변환하면 4096
hidden_nodes_num = [3000, 2000, 1000, 500, 100]        # 히든 계층의 레이어 개수를 4개로 설정, 각 계층의 노드 개수 설정
output_nodes_num = 10                         # CSV 파일에 레이블이 11개 있어서 아웃풋 레이어는 11로 설정

train_datas_num = 1110      # 10개 폰트 111글자
test_datas_num = 333        # 3개 폰트 111글자


# 신경망 초기화 및 학습
input_size = 4096  # 64x64 이미지의 픽셀 수
hidden_layers = [3000, 2000, 1000, 500, 100]  # 히든 레이어 노드 수
output_size = 10  # 클래스 수 (레이블이 10개이므로 출력 노드 수는 10)
learning_rate = 0.001
epochs = 10

#데이터 로드
def load_data(sub_dir, file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '한글글자데이터', sub_dir, file_name)

    data = []
    labels = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)


        for row in reader:
            # 첫 번째 열은 레이블, 나머지는 픽셀 데이터
            label = int(row[0]) - 1  # 라벨 값을 0~10 범위로 조정
            row_data = [float(value) for value in row[1:]]  # 픽셀 데이터

            # 데이터를 리스트에 추가
            data.append(row_data)

            # 레이블을 원-핫 인코딩 형태로 변환하여 추가
            one_hot_label = [0] * 10
            one_hot_label[label] = 1
            labels.append(one_hot_label)

    return data, labels

# 데이터 로드 및 크기 확인
train_data, train_labels = load_data('train', 'train_data.csv')
test_data, test_labels = load_data('test', 'test_data.csv')

# 데이터 준비 (예: train_data와 train_labels를 numpy 배열로 변환)
train_data = np.array(train_data)
train_labels = np.array(train_labels)

# print("훈련 데이터 개수:", len(train_data))  # 1110여야 함
# print("훈련 레이블 개수:", len(train_labels))  # 1110여야 함
# print("테스트 데이터 개수:", len(test_data))  # 333여야 함
# print("테스트 레이블 개수:", len(test_labels))  # 333여야 함

# # 첫 번째 데이터와 레이블의 크기 확인
# print("첫 번째 훈련 데이터의 크기:", len(train_data[0]))  # 4096이어야 함
# print("첫 번째 훈련 레이블의 구조:", train_labels[0])  # 원-핫 인코딩된 배열이어야 함

# # 총 픽셀 데이터 수 계산
# total_train_data_points = len(train_data) * len(train_data[0])
# total_test_data_points = len(test_data) * len(test_data[0])

# print("총 훈련 데이터 픽셀 수:", total_train_data_points)  # 예상: 4096 * 1110
# print("총 테스트 데이터 픽셀 수:", total_test_data_points)  # 예상: 4096 * 333

# 활성화 함수: 시그모이드 순전파때 사용
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수의 미분 (역전파에서 사용)
def sigmoid_derivative(x):
    return x * (1 - x)

# 손실 함수: 평균 제곱 오차를 통하여 예측값과 실제 값의 차이 측정.
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# 신경망 클래스 정의
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.001):
        # 학습률
        self.learning_rate = learning_rate
        # 가중치와 편향 초기화
        self.weights = []
        self.biases = []

        # 입력층, 히든층, 출력층 크기를 리스트로 설정
        layer_sizes = [input_size] + hidden_layers + [output_size]
        # 각 층의 가중치와 편향 초기화
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01  # 작은 난수로 초기화
            bias = np.zeros((1, layer_sizes[i + 1]))  # 편향을 0으로 초기화
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        # 순전파 수행
        self.activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, y_true):
        # 역전파 수행
        deltas = [self.activations[-1] - y_true]  # 출력층의 오차
        # 각 층에 대해 델타 계산 (오차 역전파)
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * sigmoid_derivative(self.activations[i + 1])
            deltas.append(delta)
        deltas.reverse()

        # 가중치와 편향 업데이트
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.activations[i].T.dot(deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            output = self.forward(x)
            loss = mean_squared_error(y, output)   
            self.backward(y)
            if epoch % 1 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
nn.train(train_data, train_labels, epochs)


# 현재 코드에서 에폭 로스 정확성까지 포함해서 코드 작성





