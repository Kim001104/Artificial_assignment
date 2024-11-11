import os
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
import numpy as np

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

# layer_num = 6       #총 8층의 레이어들이 있음(인,아웃풋 포함)
# epoch = 10          
# lr = 0.1            #학습률 0.1

# image_size = 64     #bmp파일의 1차원의 사이즈가 64pixel

# input_node_num = image_size*image_size      #bmp파일을 1차원 리스트로 변환하면 4096
# hidden_node_num = [1024,512,128,64]     #히든 계층의 레이어 개수를 4개로 정하고,각각 계층의 노드 개수 정해줌.
# output_nodes_num = 11                   #CSV파일에 레이블이 11개 있어서 아웃풋 레이어는 11로 설정

# train_datas_num = 1222      #11개 폰트 111글자
# test_datas_num = 333        #3개 폰트 111글자

# #활성화 함수(시그모이드, softmax)
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

# # softmax
# def softmax(input):     #마지막 출력 계층에서 계산을 용이하게 하기 위함.
#     temp = []
#     exp_value_sum = 0
#     max_value = max(input)

#     for i in range(len(input)):
#         # overflow 문제를 방지하기 위해 최댓값을 빼는 방식으로 수를 낮춘다
#         exp_value = math.exp(input[i] - max_value)
#         exp_value_sum += exp_value
#         temp.append(exp_value)
    
#     for i in range(len(input)):
#         total = exp_value_sum
#         # 각각의 값을 전체 값의 합으로 나눠준다
#         temp[i] = temp[i]/exp_value_sum

#     return temp

# #행렬 곱
# def matrix_multiplication(prev_node_num, current_node_num, weight, data):
#     output = []    
#     for i in range(current_node_num):
#         value = 0
#         for j in range(prev_node_num):
#             value += weight[i][j] * data[j]
#         output.append(value)
        
#     return output

# #퍼셉트론 정의 클래스
# class Perceptron:
#     def __init__(self, input_dim, learning_rate=0.1):
#         self.weights = random
#         pass
    
