import os

# 1. BMP 이미지 파일이 저장된 디렉토리 경로 설정
bmp_directory = r"c:\Users\motus\OneDrive\바탕 화면\인공지능\인공지능 과제2_글자\한글글자데이터\train"

# 2. 디렉토리 내부 모든 하위 디렉토리와 파일 탐색 및 BMP 파일 삭제
for root, dirs, files in os.walk(bmp_directory):
    for file in files:
        if file.endswith('.bmp'):
            file_path = os.path.join(root, file)
            os.remove(file_path)

print("모든 BMP 파일 삭제 완료!")