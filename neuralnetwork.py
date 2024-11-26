import csv
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 신경망 하이퍼파라미터 설정
input_size = 4096           # 입력 크기: 64x64 이미지의 총 픽셀 수 (4096)
hidden_layers = [250]  # 은닉층 노드 수
output_size = 111            # 출력층 노드 수: 분류할 클래스 수
learning_rate = 0.0007       # 학습률: 가중치 업데이트의 크기
epochs = 200                  # 학습 반복 횟수
batch_size = 32              # 배치 크기: 한 번에 학습할 데이터 수
drop_rate = 0.1              # 드롭아웃 비율: 은닉층의 일부 노드를 무작위로 비활성화하는 비율

# 데이터 로드 함수: CSV 파일에서 데이터를 로드하고 레이블을 원-핫 인코딩하여 반환
def load_data(sub_dir, file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 경로 가져오기
    file_path = os.path.join(current_dir, '한글글자데이터', sub_dir, file_name)  # 데이터 파일의 절대 경로 생성

    pixel_data = []      # 데이터 저장 배열
    labels = []    # 레이블 저장 배열
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0]) -1   # 첫 번째 열은 레이블 정보 (0~110로 변환)
            row_data = [float(value) / 255.0  for value in row[1:]]  # 각 픽셀 값을 0~1 범위로 정규화
            pixel_data.append(row_data)  # 픽셀 데이터를 리스트에 추가
            #원핫 인코딩 생성
            one_hot_label = [0] * output_size   #출력 크기만큼 0으로 채워진 리스트 생성
            one_hot_label[label] = 1    #해당 레이블 위치에만 1로 설정
            labels.append(one_hot_label)    #원핫 인코딩 레이블을 리스트에 추가

    return np.array(pixel_data), np.array(labels)  # 리스트를 numpy 배열로 변환하여 반환

"활성화함수 정의"

#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수: 입력값에 대한 시그모이드 함수

# Relu(순전파 사용)
def relu(x):
    return np.maximum(0, x)  # ReLU 함수: 입력이 0보다 작으면 0, 크면 그대로 출력

# Leaky Relu(순전파 사용)
def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)  # Leaky ReLU 함수: 입력이 0보다 작으면 0.01을 곱하고, 크면 그대로 출력

# tanh(순전파 사용)
def tanh(x):
    return np.tanh(x)  # tanh 함수: 입력값에 대한 쌍곡탄젠트 함수

#시그모이드 미분(역전파 사용)
def sigmoid_derivative(x):  
    return sigmoid(x) * (1 - sigmoid(x))  # 시그모이드 미분: 시그모이드 함수의 미분

# Relu 미분(역전파 사용)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # ReLU 미분: 입력이 0보다 크면 1, 작으면 0

# Leaky Rel미분
def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)  # Leaky ReLU 미분: 입력이 0보다 크면 1, 작으면 0.01

# tanh 미분(역전파 사용)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2  # tanh 미분: 1 - tanh(x)^2

# 출력층 활성화함수 정의
def softmax(x):
    exp_x = np.exp(x)  # 입력값 x에 대한 지수 함수 계산
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # softmax: 확률로 변환


"손실함수 정의"
# 평균제곱오차
def MSE(y_true, y_pred):
    return np.mean((y_true-y_pred) ** 2)
# 교차 엔트로피
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10)) / len(y_true)
    
"정확도 계산"
# 정확도 계산 함수 정의
def accuracy(y_true, y_pred):
    if y_true.ndim > 1:  
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:  
        y_pred = np.argmax(y_pred, axis=1)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)  # 정확도 계산


# "드롭아웃 함수 정의"
# # 드롭아웃 함수 정의
# def dropout(x, drop_rate):
    
#     mask = np.random.binomial(1, 1 - drop_rate, size=x.shape)  # 드롭아웃 마스크 생성
#     return x * mask / (1 - drop_rate)  # 드롭아웃 적용 후 스케일링


# 신경망 클래스 정의
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate, drop_rate):
        self.learning_rate = learning_rate  # 학습률
        self.drop_rate = drop_rate  # 드롭아웃 비율
        self.weights = []  # 가중치 저장 리스트
        self.biases = []  # 편향 저장 리스트
        
        # 네트워크 구조 정의
        layer_sizes = [input_size] + hidden_layers + [output_size]  # 입력, 은닉, 출력층 노드 수
        
        # 각 레이어 가중치와 편향 초기화
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])  # He 초기화
            bias = np.zeros((1, layer_sizes[i + 1]))  # 편향 초기화
            self.weights.append(weight)
            self.biases.append(bias)

    # 순전파
    def forward(self, x, training=True):
        self.activations = [x]  # 입력 데이터를 첫 번째 활성화 값으로 설정
        for i in range(len(self.weights) - 1):  # 은닉층에 대해 반복
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]  # 선형 계산
            a = relu(z)  # 활성화 함수 적용
            # if training:
            #     a = dropout(a, self.drop_rate)  # 드롭아웃 적용 (학습 시에만)
            self.activations.append(a)
        # 출력층 계산
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)  # 소프트맥스 활성화 함수 적용
        self.activations.append(a)
        return self.activations[-1]

    # 역전파
    def backward(self, y_true):
        # 출력층 오차
        deltas = [self.activations[-1] - y_true]
        
        # 은닉층에서의 오차 전파
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * relu_derivative(self.activations[i + 1])
            deltas.append(delta)
        deltas.reverse()  # 순서를 원래대로 변경
        
        # 가중치 및 편향 업데이트
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.activations[i].T.dot(deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    # 학습
    def train(self, x, y, test_data, test_labels, epochs, learning_rate):
        self.learning_rate = learning_rate  # 학습률 설정
        for epoch in range(epochs):  # 에포크 반복
            if (epoch + 1) % 50 == 0:  # 50 에포크마다 학습률 감소
                self.learning_rate *= 0.5

            # 순전파 및 손실 계산
            output = self.forward(x, training=True)
            loss = cross_entropy(y, output)

            # 정확도 계산
            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            train_accuracy = np.sum(predictions == true_labels) / len(y) * 100

            # 역전파 수행
            self.backward(y)

            # 테스트 데이터 평가
            test_output = self.forward(test_data, training=False)
            test_loss = cross_entropy(test_labels, test_output)
            test_predictions = np.argmax(test_output, axis=1)
            test_true_labels = np.argmax(test_labels, axis=1)
            test_accuracy = accuracy(test_true_labels, test_predictions)

            # 에포크 결과 출력
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")

    # 예측
    def predict(self, x):
        output = self.forward(x, training=False)
        return np.argmax(output, axis=1)


def img_predict(nn, train_data,test_data,train_labels, test_labels,title="Train vs Test (True vs Predicted Images)"):
    # 임의로 10개의 인덱스를 선택하여 학습 데이터와 테스트 데이터에서 이미지와 레이블을 가져옵니다.
    indices = np.random.choice(len(train_data), 10, replace=False)

    plt.figure(figsize=(20, 5))

    for i, idx in enumerate(indices):
        train_image = train_data[idx].reshape(64, 64)  # 학습 데이터의 64x64 크기 이미지
        true_label = np.argmax(train_labels[idx])      # 학습 데이터의 실제 레이블
        

        predicted_label = nn.predict(train_data[idx:idx + 1])[0]  # 학습 데이터 예측

        # 테스트 데이터에서 동일한 레이블을 찾음
        test_idx = next((j for j, label in enumerate(test_labels) if nn.predict(test_data[j:j+1])[0]==predicted_label), None)

        if test_idx is not None:
            test_image = test_data[test_idx].reshape(64, 64)  # 테스트 데이터의 예측 이미지 (노이즈 포함)

        else:
            print(f"No test image found for label {true_label}")
            continue

        # 학습 이미지와 예측된 테스트 이미지 표시
        plt.subplot(2, 10, i * 2 + 1)
        plt.imshow(train_image, cmap='gray')
        plt.title(f"Train True: {true_label}")
        plt.axis('off')

        plt.subplot(2, 10, i * 2 + 2)
        plt.imshow(test_image, cmap='gray')
        plt.title(f"Test Pred: {predicted_label}")
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.show()

# 데이터 로드
train_data, train_labels = load_data('train3', 'train3_data.csv')
test_data, test_labels = load_data('test', 'test_data.csv')

# 신경망 객체 생성 및 학습 수행
nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate,drop_rate)

nn.train(train_data, train_labels, test_data, test_labels, epochs, learning_rate)

# 학습 데이터에서 10개의 이미지 쌍을 시각화하여 실제 레이블과 예측 레이블의 이미지를 비교
img_predict(nn, train_data, test_data,train_labels,test_labels, title="Train vs Test (True vs Predicted Images)")






