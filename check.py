##데이터 잘 뽑혔나 확인용
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# 1번 레이블 파일 열기
label = "1"
bmp_folder_path = "C:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\train\\가"
file_path = os.path.join(bmp_folder_path, f"{label}.bmp")

# 이미지 파일 열기 및 픽셀 데이터 추출
img = Image.open(file_path).convert('L')  # 흑백 모드로 변환
pixel_data = list(img.getdata())  # 1차원 픽셀 데이터 추출

# 이미지의 크기(너비, 높이) 가져오기
width, height = img.size

# numpy 배열로 1차원 배열을 2차원으로 변환
pixel_array_2d = np.array(pixel_data).reshape((height, width))

# numpy 배열을 이미지로 변환
plt.imshow(pixel_array_2d, cmap='gray', vmin=0, vmax=255)
plt.show()