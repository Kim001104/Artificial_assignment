import csv
import os
import numpy as np
from PIL import Image

# 신경망 하이퍼파라미터 설정
input_size = 4096           # 입력 크기: 64x64 이미지의 총 픽셀 수 (4096)
hidden_layers = [150]  # 은닉층 노드 수
output_size = 111            # 출력층 노드 수: 분류할 클래스 수
learning_rate = 0.0001       # 학습률: 가중치 업데이트의 크기
epochs = 200                  # 학습 반복 횟수
batch_size = 32              # 배치 크기: 한 번에 학습할 데이터 수
drop_rate = 0.5              # 드롭아웃 비율: 은닉층의 일부 노드를 무작위로 비활성화하는 비율

# 데이터 경로 설정
base_dir = "C:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터"
sub_dir = "train3"

target_dir = os.path.join(base_dir, sub_dir)
print("Target directory:", target_dir)  # 확인


def load_data(base_dir, sub_dir, output_size=111):
    """
    특정 디렉토리 내의 모든 BMP 파일을 읽고, 데이터와 레이블을 반환하는 함수.

    Args:
    - base_dir (str): 데이터가 위치한 최상위 경로 (예: '한글글자데이터').
    - sub_dir (str): base_dir 내 하위 폴더 이름 (예: 'train3').
    - output_size (int): 원-핫 인코딩을 위한 출력 크기 (클래스 수).

    Returns:
    - pixel_data (np.array): 모든 BMP 이미지 데이터를 병합한 배열.
    - labels (np.array): 모든 레이블을 원-핫 인코딩한 배열.
    """
    # train3 디렉토리 경로 생성
    target_dir = os.path.join(base_dir, sub_dir)

    pixel_data = []  # BMP 이미지 데이터를 저장할 리스트
    labels = []      # 레이블 데이터를 저장할 리스트
    label_mapping = {}  # 폴더 이름 → 고유 레이블 매핑

    # 현재 레이블 번호
    current_label = 0

    # train3 하위 디렉토리 순회 (예: 가, 나, ...)
    for folder_name in os.listdir(target_dir):
        folder_path = os.path.join(target_dir, folder_name)

        # 하위 폴더가 글자 이름일 경우만 처리
        if os.path.isdir(folder_path):
            if folder_name not in label_mapping:
                label_mapping[folder_name] = current_label
                current_label += 1

            # 레이블 가져오기
            label = label_mapping[folder_name]

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".bmp"):  # BMP 파일만 처리
                    file_path = os.path.join(folder_path, file_name)

                    # BMP 이미지 읽기
                    img = Image.open(file_path).convert('L')  # 흑백 모드로 변환
                    img_array = np.array(img).flatten()  # 2D 이미지를 1D로 펼침
                    img_normalized = img_array / 255.0  # 픽셀 값을 0~1 범위로 정규화

                    pixel_data.append(img_normalized)  # 이미지 데이터를 추가

                    # 원-핫 인코딩 생성
                    if output_size is None:
                        output_size = len(label_mapping)
                    one_hot_label = [0] * output_size
                    one_hot_label[label] = 1
                    labels.append(one_hot_label)  # 레이블 데이터를 추가

    # 리스트를 numpy 배열로 변환 후 반환
    return np.array(pixel_data), np.array(labels), label_mapping

"활성화함수 정의"

# Relu(순전파 사용)
def relu(x):
    return np.maximum(0, x)  # ReLU 함수: 입력이 0보다 작으면 0, 크면 그대로 출력

# tanh(순전파 사용)
def tanh(x):
    return np.tanh(x)  # tanh 함수: 입력값에 대한 쌍곡탄젠트 함수

# Relu 미분(역전파 사용)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # ReLU 미분: 입력이 0보다 크면 1, 작으면 0

# tanh 미분(역전파 사용)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2  # tanh 미분: 1 - tanh(x)^2

# 출력층 활성화함수 정의
def softmax(x):
    exp_x = np.exp(x)  # 입력값 x에 대한 지수 함수 계산
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # softmax: 확률로 변환

# 손실함수 정의
def MSE(y_true, y_pred):
    return np.mean((y_true-y_pred) ** 2)
    
"정확도 계산"
# 정확도 계산 함수 정의
def accuracy(y_true, y_pred):
    if y_true.ndim > 1:  
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:  
        y_pred = np.argmax(y_pred, axis=1)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)  # 정확도 계산
"드롭아웃 함수 정의"
# 드롭아웃 함수 정의
def dropout(x, drop_rate):
    """드롭아웃 함수: 노드의 일부를 무작위로 비활성화"""
    mask = np.random.binomial(1, 1 - drop_rate, size=x.shape)  # 드롭아웃 마스크 생성
    return x * mask / (1 - drop_rate)  # 드롭아웃 적용 후 스케일링


# 신경망 클래스 정의
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate,drop_rate):  # 초기화 함수
        self.learning_rate = learning_rate  # 학습률
        self.drop_rate = drop_rate  # 드롭아웃 비율
        self.weights = []   # 가중치 저장 리스트
        self.biases = []    # 편향 저장 리스트
        layer_sizes = [input_size] + hidden_layers + [output_size]  # 각 레이어의 노드 수

        # 각 레이어의 가중치와 편향 초기화
        for i in range(len(layer_sizes) - 1):   # 각 레이어에 대해 가중치와 편향 초기화
            # weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])    # He 초기화
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1.0 / layer_sizes[i])    # Xavier 초기화
            bias = np.zeros((1, layer_sizes[i + 1]))    # 편향 초기화
            self.weights.append(weight) # 가중치 추가
            self.biases.append(bias)    # 편향 추가

    # 순전파 : 입력 데이터를 각 레이어에 전달하여 최종 출력값 계산
    def forward(self, x, training=True):
        self.activations = [x]  # 입력 데이터 추가
        for i in range(len(self.weights) - 1):  # 은닉층의 활성화 함수 적용
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            if training:
                a = dropout(a, self.drop_rate)  
            self.activations.append(a)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        self.activations.append(a)
        return self.activations[-1]

    # 역전파 : 순전파의 최종 출력값과 실제 레이블 간의 차이를 계산하여 손실 측정
    def backward(self, y_true): 
        deltas = [self.activations[-1] - y_true]    # 출력층의 오차
        for i in reversed(range(len(self.weights) - 1)):    # 역방향으로 오차 전파
            delta = deltas[-1].dot(self.weights[i + 1].T) * relu_derivative(self.activations[i + 1])    # 은닉층의 오차
            deltas.append(delta)    # 오차 저장
        deltas.reverse()    # 오차 역순으로 정렬

        for i in range(len(self.weights)):  # 가중치 및 편향 업데이트
            self.weights[i] -= self.learning_rate * self.activations[i].T.dot(deltas[i])    # 가중치 업데이트
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)   # 편향 업데이트

    def train(self, x, y, test_data, test_labels, epochs, batch_size,learning_rate=0.0005):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            if (epoch + 1) % 100 == 0:  # 100번째 에포크마다 학습률을 절반으로 줄임
                self.learning_rate *= 0.5   # 학습률 감소
            
            indices = np.arange(x.shape[0]) # 데이터 인덱스 생성
            np.random.shuffle(indices)  # 인덱스 섞기
            total_loss = 0  # 총 손실 초기화
            correct_predictions = 0 # 정확한 예측 수 초기화

            for start_idx in range(0, x.shape[0], batch_size):  # 배치 단위로 학습
                batch_indices = indices[start_idx:start_idx + batch_size]   # 배치 인덱스 선택
                batch_x = x[batch_indices]  # 배치 데이터
                batch_y = y[batch_indices]  # 배치 레이블

                output = self.forward(batch_x, training = True)  # 순전파 수행
                loss = MSE(batch_y, output) # 손실 계산
                total_loss += loss  # 총 손실 누적

                predictions = np.argmax(output, axis=1) # 예측값
                true_labels = np.argmax(batch_y, axis=1)    # 실제 레이블
                correct_predictions += np.sum(predictions == true_labels)   # 정확한 예측 수

                self.backward(batch_y)  # 역전파 수행

            # train_accuracy = correct_predictions / x.shape[0] * 100 # 정확도 계산
            average_loss = total_loss / (x.shape[0] // batch_size)  # 평균 손실 계산

            # 테스트 데이터 정확도 계산
            test_output = self.forward(test_data)
            test_loss = MSE(test_labels, test_output)
            test_predictions = np.argmax(test_output, axis=1)
            test_true_labels = np.argmax(test_labels, axis=1)
            test_accuracy = accuracy(test_true_labels, test_predictions)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss:.4f},Test Loss : {test_loss:.4f},Test Accuracy: {test_accuracy * 100:.2f}%")

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)


import matplotlib.pyplot as plt
import numpy as np

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
train_data, train_labels = load_data(base_dir, 'train3', output_size=111)
test_data, test_labels = load_data('test', 'test_data.csv',output_size=111)

# 신경망 객체 생성 및 학습 수행
nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate,drop_rate)
nn.train(train_data, train_labels, test_data, test_labels, epochs, batch_size,learning_rate)

# 학습 데이터에서 10개의 이미지 쌍을 시각화하여 실제 레이블과 예측 레이블의 이미지를 비교
img_predict(nn, train_data, test_data,train_labels,test_labels, title="Train vs Test (True vs Predicted Images)")
