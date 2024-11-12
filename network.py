import csv
import random
import math
import os
import numpy as np  
# import matplotlib.pyplot as plt


# total_layers_num = 7       # 총 8층의 레이어들이 있음(인,아웃풋 포함)
# epoch = 10          
# lr = 0.1            # 학습률 0.1

# image_size = 64     # bmp 파일의 1차원의 사이즈가 64 픽셀

# input_node_num = image_size * image_size     # bmp 파일을 1차원 리스트로 변환하면 4096
# hidden_nodes_num = [3000, 2000, 1000, 500, 100]        # 히든 계층의 레이어 개수를 4개로 설정, 각 계층의 노드 개수 설정
# output_nodes_num = 10                         # CSV 파일에 레이블이 11개 있어서 아웃풋 레이어는 11로 설정

# train_datas_num = 1110      # 10개 폰트 111글자
# test_datas_num = 333        # 3개 폰트 111글자

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

# 신경망 초기화 및 학습


#편향은 b로 설정하고 뉴런이 얼마나 "쉽게 활성화" 되느냐를 제어함.
#반면 w1,w2는 가중치로 각 신호에 얼마나 많은 "영향"을 주느냐
#활성화 함수: 입력신호의 총합이 활성화를 일으키는지 정하는 역할
# a는 입력신호의 총합 h()는 활성화 함수, y는 출력
#활성화 함수는 계단 함수로도 불리는데 "시그모이드 함수" 사용


#기계학습이란? 데이터에서 답을 찾고 패턴을 발견하여 데이터로 이야기를 만드는 것!!!
#특징을 추출하고 그 특징의 패턴을 기계학습 기술로 학습하는 방법이 있다. 여기에서 말하는 특징이란 
#입력 데이터(폰트 글씨들.bmp)에서 본질적인 데이터(픽셀의 검은색과 하얀색)을 정확하게 추출할 수 있도록 설계된 변환기
#훈련 데이터와 시험 데이터를 나누는 이유느?
#범용능력에 있어서 인데 왜??(아직 보지 못 하였던 문제를 올바르게 찾아야함.)
#그렇다면 폰트를 컴퓨터가 인식을 해야하는데 아직 한번도 보지 못하였으니깐 데이터에서 훈련 데이터 따로 테스트 데이터로 나눠야한다는 소리
input_size = 4096  # 64x64 이미지의 픽셀 수
hidden_layers = [2000,1000]  # 히든 레이어 노드 수
output_size = 110  # 클래스 수 (레이블이 10개이므로 출력 노드 수는 10)
learning_rate = 0.001
epochs = 3  

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
            label = int(row[0])   # 라벨 값을 0~10 범위로 조정
            row_data = [float(value) for value in row[1:]]  # 픽셀 데이터

            # 데이터를 리스트에 추가
            data.append(row_data)

            # 레이블을 원-핫 인코딩 형태로 변환하여 추가
            one_hot_label = [0] * 110
            one_hot_label[label-1] = 1
            labels.append(one_hot_label)

    return data, labels

# 데이터 로드 및 크기 확인
train_data, train_labels = load_data('train', 'train_data.csv')
test_data, test_labels = load_data('test', 'test_data.csv')

# 데이터 준비 (예: train_data와 train_labels를 numpy 배열로 변환)
train_data = np.array(train_data)
train_labels = np.array(train_labels)
#데이터를 0혹은 1로 정규화 시켜줌.



# # 활성화 함수: 시그모이드
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # 시그모이드의 미분
# def sigmoid_derivative(x):
#     return x * (1 - x)

# ReLU 활성화 함수
def relu(x):
    return np.maximum(0, x)

# ReLU의 미분 함수
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 마지막 계층 소프트맥스 함수
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 손실 함수: 평균 제곱 오차
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# 정확도 계산
def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

use_mse = True

def loss_function(y_true, y_pred):
    if use_mse:
        return mean_squared_error(y_true, y_pred)
    else:
        return categorical_crossentropy(y_true, y_pred)

# 각 층을 담당하는 Layer 클래스
# class Layer:
#     def __init__(self, input_size, output_size, learning_rate=0.001):
#         # 가중치와 바이어스 초기화
#         self.weights = np.random.randn(input_size, output_size) * 0.01
#         self.biases = np.zeros((1, output_size))
#         self.learning_rate = learning_rate

class Layer:
    def __init__(self, input_size, output_size, learning_rate):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)  # He 초기화
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    # 순전파
    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.output = relu(self.z)  # ReLU 활성화 함수 사용
        return self.output

    # 역전파
    def backward(self, output_error):
        # 델타 계산: 오차의 기울기 = 이전 층에서 전달된 오차 * 현재 층의 활성화 함수 미분값
        self.delta = output_error * relu_derivative(self.output)

        # 가중치와 바이어스 기울기 계산
        weight_gradient = np.dot(self.input.T, self.delta)
        bias_gradient = np.sum(self.delta, axis=0, keepdims=True)

        # 가중치와 바이어스 업데이트
        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient

        # 이전 층으로 전달할 오차 계산
        input_error = np.dot(self.delta, self.weights.T)
        return input_error

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate):
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], learning_rate))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = softmax(layer.forward(x))  # 출력층에 softmax 적용
            else:
                x = layer.forward(x)
        return x

    def backward(self, y_true):
        output_error = self.layers[-1].output - y_true
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error)

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            output = self.forward(x)
            loss = loss_function(y, output)
            self.backward(y)
            acc = accuracy(y, output)
            print(f"Epoch {epoch + 1}, Loss: {loss}, Accuracy: {acc * 100:.2f}%")

# train_labels의 형태를 출력하여 차원이 맞는지 확인
print("train_labels shape:", train_labels.shape)



# 신경망 생성 및 학습
nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
nn.train(train_data, train_labels, epochs)
