import csv
import os
import math
import random
import numpy as np

# 신경망 하이퍼파라미터 설정
input_size = 4096           # 입력 크기: 64x64 이미지의 총 픽셀 수 (4096)
output_size = 110           # 출력층 노드 수: 분류할 클래스 수
hidden_layers = [1000, 500] # 은닉층 노드 수
lr = 0.001                  # 학습률  
epochs = 3                  # 전체 데이터셋에 대해 학습 반복 횟수

# 데이터 로드 함수: CSV 파일에서 데이터를 로드하고 레이블을 원-핫 인코딩하여 반환
def load_data(sub_dir, file_name):
    current_dir = os.getcwd()  # 현재 디렉토리 가져오기
    file_path = os.path.join(current_dir, '한글글자데이터', sub_dir, file_name)  # 데이터 파일의 절대 경로 생성

    data = []      # 데이터 저장 리스트
    labels = []    # 레이블 저장 리스트
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0]) - 1  # 첫 번째 열은 레이블 정보 (0~109로 변환)
            row_data = [float(value) / 255.0 for value in row[1:]]  # 데이터 정규화 (0~1 범위)
            data.append(row_data)

            # 레이블을 원-핫 인코딩하여 추가
            one_hot_label = [0] * output_size
            one_hot_label[label] = 1
            labels.append(one_hot_label)

    return np.array(data), np.array(labels)

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 가중치 및 편향 초기화
        self.weights = []
        self.biases = []

        # 입력층과 첫 번째 은닉층
        self.weights.append(np.random.uniform(-0.05, 0.05, (input_size, hidden_layers[0])))
        self.biases.append(np.zeros(hidden_layers[0]))

        # 은닉층들
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.uniform(-0.05, 0.05, (hidden_layers[i - 1], hidden_layers[i])))
            self.biases.append(np.zeros(hidden_layers[i]))

        # 마지막 은닉층과 출력층
        self.weights.append(np.random.uniform(-0.05, 0.05, (hidden_layers[-1], output_size)))
        self.biases.append(np.zeros(output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def MSE(self, output, target):
        return np.mean((output - target) ** 2)

    def forward(self, input_data):
        """
        순전파 수행.
        입력 데이터를 각 레이어를 통과시켜 활성화 결과를 반환.
        """
        activations = [input_data]
        for i in range(len(self.weights) - 1):  # 은닉층
            input_data = self.relu(np.dot(input_data, self.weights[i]) + self.biases[i])
            activations.append(input_data)
        # 출력층
        input_data = self.sigmoid(np.dot(input_data, self.weights[-1]) + self.biases[-1])
        activations.append(input_data)
        return activations

    def backward(self, activations, target):
        """
        역전파 수행.
        활성화 값과 타겟 값을 사용하여 가중치 및 편향의 기울기를 계산.
        """
        gradients_w = []
        gradients_b = []

        # 출력층 오차
        delta = activations[-1] - target
        gradients_w.append(np.dot(activations[-2].T, delta))
        gradients_b.append(np.sum(delta, axis=0))

        # 은닉층 역전파
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.relu_derivative(activations[i + 1])
            gradients_w.append(np.dot(activations[i].T, delta))
            gradients_b.append(np.sum(delta, axis=0))

        # 순서 뒤집기
        gradients_w.reverse()
        gradients_b.reverse()

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """
        계산된 기울기를 사용하여 가중치와 편향 업데이트.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, train_data, train_labels, epochs):

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0


            for i in range(len(train_data)):

                activations = self.forward(train_data[i:i+1])  # 입력 데이터의 1개 샘플
                output = activations[-1]

                total_loss += self.MSE(output, train_labels[i:i+1])

                # 정확도 계산
                if np.argmax(output) == np.argmax(train_labels[i]):
                    correct_predictions += 1
                
                # 역전파
                gradients_w, gradients_b = self.backward(activations, train_labels[i:i+1])
                self.update_weights(gradients_w, gradients_b)

                # 결과 출력
                accuracy = correct_predictions / len(train_data) * 100
                average_loss = total_loss / len(train_data)
                print(f"\r에폭 {epoch+1}/{epochs}, 손실: {average_loss:.4f}, 정확도: {accuracy:.2f}%", end="")


    # 평가용 
    def evaluate(self, test_data, test_labels):
        total_loss = 0
        correct_predictions = 0

        for i in range(len(test_data)):
            # 순전파
            activations = self.forward(test_data[i:i+1])
            output_size = (activations[-1])

            # 손실 계산
            total_loss += self.MSE(output_size, test_labels[i:i+1])

            # 정확도 계산
            if np.argmax(output_size) == np.argmax(test_labels[i]):
                correct_predictions += 1

        test_accuracy = correct_predictions / len(test_data) * 100
        average_loss = total_loss / len(test_data)
        print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        return average_loss, test_accuracy


# 데이터 로드
train_data, train_labels = load_data('train', 'train_data.csv')
test_data, test_labels = load_data('test', 'test_data.csv')

nn = NeuralNetwork(input_size, hidden_layers, output_size, lr)

# 학습
print("Training the model...")
nn.train(train_data, train_labels, epochs)

# 평가
print("\nEvaluating the model on test data...")
nn.evaluate(test_data, test_labels)
