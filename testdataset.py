from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import random

#주어진 문장들
sentences = [
    "경 기 도 고 양 시 일 산 서 구 탄 현 로 1 3 6 1 1 2 동 1 6 0 1 호",
    "김 동 현",
    "김 재 규 이 경 숙 김 효 섭 박 노 쇠",
    "인 천 광 역 시 연 수 구 아 카 데 미 로 1 1 9",
    "청 춘 이 는 듣 기 만 하 여 도 가 슴 이 설 레 는 말 이 다 너 의 두 손을 가 슴 에 대 고 물 방 아 같 은 심 장 의 고 동 을 들 어 보 라",
    "청 춘 의 피 는 끓 는 다 피 에 뛰 노 는 심 장 은 거 선 의 기 관 같 이 힘 있 다 이 것 이 다 인 류 의 역 사 를 꾸 며 내 려 온 동 력 은 바 로 이 것 이 다",
    "이 성 은 투 명 하 되 얼 음 과 같 으 며 지 혜 는 날 카 로 우 나 갑 속 에 든 칼 이 다 청 춘 의 끓 는 피 가 아 니 더 면 인 간 이 얼 마 나 쓸 쓸 하 랴",
    "얼 음 에 싸 인 만 물 은 죽 음 이 있 을 뿐 이 다"
]

#고유한 글자 추출
unique_chars = set()
for sentence in sentences:
    for char in sentence:
        if char.strip() and char.isalnum():  # 공백 및 특수문자 제외
            unique_chars.add(char)

print(f"총 고유 글자 수: {len(unique_chars)}")

#이미지 저장 디렉토리 설정
base_output_dir = "c:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\test"
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

#폰트 경로 설정
font_paths = [
    "C:\\Windows\\Fonts\\ahn_l.ttf",    # 안상수2006가는보통
    "C:\\Windows\\Fonts\\HMFMOLD.TTF", # 휴먼옛체
    "C:\\Windows\\Fonts\\HMFMPYUN.TTF"  # 휴먼편지체
]

#이미지 생성 및 변형 함수
def create_and_augment_image(char, font_path, output_path):
    # 글자 이미지 생성
    font = ImageFont.truetype(font_path, 64)  # 폰트 크기 설정
    image = Image.new("L", (64, 64), "white")  # 흰 배경의 64x64 이미지 생성
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textsize(char, font=font)
    draw.text(((64 - text_width) // 2, (64 - text_height) // 2), char, font=font, fill="black")

    # 변형 적용 함수
    image = augment_image(image)
    
    # 이미지 저장
    image.save(output_path, format="bmp")

#변형 함수 (가로/세로 노이즈 추가)
def augment_image(image):
    
    transformations = "add_noise"

    for transform in transformations:
        if transform == "add_noise":
            image_cv = np.array(image)
            noise = np.zeros_like(image_cv)

            # 가로 또는 세로 방향 중 하나 선택
            noise_type = random.choice(["horizontal", "vertical"])

            if noise_type == "horizontal":
                for i in range(0, image_cv.shape[0], 5):  # 몇몇 행에만 노이즈 추가
                    noise[i, :] = np.random.normal(0, 10, image_cv.shape[1])
            elif noise_type == "vertical":
                for j in range(0, image_cv.shape[1], 5):  # 몇몇 열에만 노이즈 추가
                    noise[:, j] = np.random.normal(0, 10, image_cv.shape[0])

            image_cv = cv2.add(image_cv, noise.astype(np.uint8))
            image = Image.fromarray(image_cv)

    return image

#각 글자마다 디렉토리 생성 및 이미지 저장
for char in unique_chars:
    char_dir = os.path.join(base_output_dir, char)
    os.makedirs(char_dir, exist_ok=True)

    # 파일 번호 초기화
    file_index = 1

    for font_path in font_paths:
        output_path = os.path.join(char_dir, f"{file_index}.bmp")
        create_and_augment_image(char, font_path, output_path)
        
        # 인덱스를 증가시켜 다음 이미지 파일 이름으로 사용
        file_index += 1

print("완료")
