import os
import csv
import math
import random
from PIL import Image

# 하이퍼파라미터 설정
input_size = 4096      # 64x64 이미지의 입력 크기
hidden_layers = [96]   # 히든 레이어 노드 수
output_size = 110      # 클래스 개수
learning_rate = 0.01
epochs = 3

# 데이터 로드 함수
def load_data(sub_dir, file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '한글글자데이터', sub_dir, file_name)

    data = []
    labels = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            label = int(row[0])  # 레이블 값
            row_data = [float(value) for value in row[1:]]  # 픽셀 데이터
            data.append(row_data)

            one_hot_label = [0] * output_size
            one_hot_label[label] = 1
            labels.append(one_hot_label)

    return data, labels

# ReLU 함수 및 Softmax 함수
def relu(input):
    return max(input, 0)

def all_relu(input):
    return [relu(value) for value in input]

def softmax(input):
    max_value = max(input)
    exp_values = [math.exp(i - max_value) for i in input]
    sum_exp = sum(exp_values)
    return [value / sum_exp for value in exp_values]

# 행렬 곱셈 함수
def matrix_multiplication(prev_node_num, current_node_num, weight, data):
    output = []
    for i in range(current_node_num):
        value = sum(weight[i][j] * data[j] for j in range(prev_node_num))
        output.append(value)
    return output

# Layer 클래스 정의
class Layer:
    def __init__(self, act_type, layer_idx):
        self.act = act_type
        self.layer_idx = layer_idx
        self.weight = []
        self.bias = []
        self.cache = {}
        self.dW = []
        self.db = []

    def init_parameter(self, current_node_num, prev_node_num, layer_num):
        stddev = math.sqrt(2.0 / prev_node_num) if self.layer_idx != layer_num - 1 else math.sqrt(1.0 / prev_node_num)
        weight = [[random.gauss(0, stddev) for _ in range(prev_node_num)] for _ in range(current_node_num)]
        bias = [0 for _ in range(current_node_num)]
        dW = [[0 for _ in range(prev_node_num)] for _ in range(current_node_num)]
        db = [0 for _ in range(current_node_num)]
        return weight, bias, dW, db

    def save_set(self, prev_act, Z):
        self.cache['prev_act'] = prev_act
        self.cache['Z'] = Z

    def save_grad(self, dW, db, current_node_num, prev_node_num):
        for i in range(current_node_num):
            self.db[i] += db[i]
            for j in range(prev_node_num):
                self.dW[i][j] += dW[i][j]

    def update_parameter(self, current_node_num, prev_node_num, lr, train_data_num):
        for i in range(current_node_num):
            self.db[i] /= train_data_num
            for j in range(prev_node_num):
                self.dW[i][j] /= train_data_num
        for i in range(current_node_num):
            self.bias[i] -= self.db[i] * lr
            for j in range(prev_node_num):
                self.weight[i][j] -= lr * self.dW[i][j]

    def clear_grad(self, current_node_num, prev_node_num):
        for i in range(current_node_num):
            self.db[i] = 0
            for j in range(prev_node_num):
                self.dW[i][j] = 0

# ModelStruct 클래스 정의
class ModelStruct:
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.layers = []
        self.layer_node_num = []
        self.train_data = []
        self.test_data = []
        self.train_target = []
        self.test_target = []
        self.train_data_num = 0
        self.test_data_num = 0

    def load_data(self, train_sub_dir, train_file_name, test_sub_dir, test_file_name):
        self.train_data, self.train_target = load_data(train_sub_dir, train_file_name)
        self.test_data, self.test_target = load_data(test_sub_dir, test_file_name)
        self.train_data_num = len(self.train_data)
        self.test_data_num = len(self.test_data)

    def insert_new_layer(self, act_type, layer_idx):
        layer = Layer(act_type, layer_idx)
        self.layers.append(layer)
        if act_type != 'none':
            self.layers[layer_idx].weight, self.layers[layer_idx].bias, self.layers[layer_idx].dW, self.layers[layer_idx].db = layer.init_parameter(
                current_node_num=self.layer_node_num[layer_idx],
                prev_node_num=self.layer_node_num[layer_idx - 1],
                layer_num=self.layer_num
            )

    def forward_pass(self, data_idx, data_type):
        act = self.train_data[data_idx] if data_type == "train" else self.test_data[data_idx]
        for i in range(1, self.layer_num):
            Z = matrix_multiplication(
                prev_node_num=self.layer_node_num[i-1],
                current_node_num=self.layer_node_num[i],
                weight=self.layers[i].weight,
                data=act
            )
            for j in range(self.layer_node_num[i]):
                Z[j] += self.layers[i].bias[j]
            self.layers[i].save_set(act, Z)
            act = all_relu(Z) if self.layers[i].act == "relu" else softmax(Z)
        return act

    def calc_output_layer_dZ(self, output, target):
        return [output[i] - target[i] for i in range(len(output))]

    def backward_pass(self, output, data_idx):
        dZ = self.calc_output_layer_dZ(output, self.train_target[data_idx])
        dW = self.calc_dW(self.layer_node_num[-1], self.layer_node_num[-2], dZ, -1)
        db = dZ
        dA_prev = self.calc_dA_prev(self.layer_node_num[-1], self.layer_node_num[-2], -1, dZ)
        self.layers[-1].save_grad(dW, db, self.layer_node_num[-1], self.layer_node_num[-2])

        for i in range(self.layer_num - 2, 0, -1):
            dZ = self.calc_hidden_layer_dZ(dA_prev, i)
            dW = self.calc_dW(self.layer_node_num[i], self.layer_node_num[i-1], dZ, i)
            db = dZ
            dA_prev = self.calc_dA_prev(self.layer_node_num[i], self.layer_node_num[i-1], i, dZ)
            self.layers[i].save_grad(dW, db, self.layer_node_num[i], self.layer_node_num[i-1])

    def train(self):
        for e in range(epochs):
            correct = sum(1 for idx in range(self.train_data_num) if self.isCorrect(self.forward_pass(idx, "train"), idx))
            print(f"Epoch {e+1}/{epochs}, Accuracy: {correct / self.train_data_num}")
            for i in range(1, self.layer_num):
                self.layers[i].update_parameter(self.layer_node_num[i], self.layer_node_num[i-1], learning_rate, self.train_data_num)
                self.layers[i].clear_grad(self.layer_node_num[i], self.layer_node_num[i-1])

    def predict(self):
        correct = sum(1 for idx in range(self.test_data_num) if self.isCorrect(self.forward_pass(idx, "test"), idx))
        print(f"Test Accuracy: {correct / self.test_data_num}")

    def isCorrect(self, output, data_idx):
        return output.index(max(output)) == self.train_target[data_idx].index(1)

# 모델 초기화 및 레이어 추가
model = ModelStruct(layer_num=3)
model.layer_node_num = [input_size] + hidden_layers + [output_size]

for i in range(model.layer_num):
    if i == 0:
        model.insert_new_layer(act_type="none", layer_idx=i)
    elif i == model.layer_num - 1:
        model.insert_new_layer(act_type="softmax", layer_idx=i)
    else:
        model.insert_new_layer(act_type="relu", layer_idx=i)

# 데이터 로드 및 모델 훈련
train_sub_dir = 'train'
train_file_name = 'train_data.csv'
test_sub_dir = 'test'
test_file_name = 'test_data.csv'
model.load_data(train_sub_dir, train_file_name, test_sub_dir, test_file_name)
model.train()
model.predict()
