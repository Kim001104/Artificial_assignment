from PIL import Image, ImageDraw, ImageFont
import os

# 1. 주어진 문장들
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

# 2. 고유한 글자 추출
unique_chars = set()
for sentence in sentences:
    for char in sentence:
        # 공백 및 특수문자 제외 (필요 시 포함 가능)
        if char.strip() and char.isalnum():
            unique_chars.add(char)

print(f"총 고유 글자 수: {len(unique_chars)}")

# 3. 이미지 저장 디렉토리 설정
base_output_dir = "c:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\train"
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# 4. 폰트 파일 경로 설정
# 여러 폰트를 리스트로 설정
font_paths = [
    "C:\\Windows\\Fonts\\gulim.ttc",
    "C:\\Windows\\Fonts\\NGULIM.TTF",
    "C:\\Windows\\Fonts\\HMFMPYUN.TTF",
    "C:\\Windows\\Fonts\\HMKMAMI.TTF",
    "C:\\Windows\\Fonts\\HMFMMUEX.TTC",
    "C:\\Windows\\Fonts\\UNI_HSR.TTF",
    "C:\\Windows\\Fonts\\HYGTRE.TTF",
    "C:\\Windows\\Fonts\\HYMJRE.TTF",
    "C:\\Windows\\Fonts\\HYGPRM.TTF",
    "C:\\Windows\\Fonts\\HYWULB.TTF"
]

# 5. 글자당 이미지 생성 함수
def create_char_image(char, font_path, save_dir, img_size=(64, 64), font_size=64): 
        # 흰색 배경으로 이미지 생성 ('L' 모드: 흑백)
        img = Image.new('L', img_size, color=255)  # 배경 흰색
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=font_size)
        
        # 글자의 크기 계산
        width, height = draw.textsize(char, font=font)
        
        # 글자를 중앙에 배치
        position = ((img_size[0]-width)/2, (img_size[1]-height)/2)
        draw.text(position, char, fill=0, font=font)  # 글자 검은색 
        
        # 폰트 이름 추출 (파일명에서 확장자 제거)
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        
        # 이미지 파일명에 글자와 폰트 이름 포함 (예: '경_gulim.bmp')
        img_filename = f"{char}_{font_name}.bmp"
        img_path = os.path.join(save_dir, img_filename)
        img.save(img_path)
        
# 6. 모든 고유 글자에 대해 이미지 생성
for char in unique_chars:
    # 글자별 디렉토리 경로 설정
    char_dir = os.path.join(base_output_dir, char)
    if not os.path.exists(char_dir):
        os.makedirs(char_dir)
    
    # 각 폰트별로 이미지 생성 및 저장
    for font_path in font_paths:
        create_char_image(char, font_path, char_dir)
        
print("모든 글자의 BMP 이미지 생성 완료!")



