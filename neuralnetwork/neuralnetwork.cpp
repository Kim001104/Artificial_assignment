#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 4096  // CSV 파일 한 줄의 최대 길이
#define OUTPUT_SIZE 111       // 원-핫 인코딩 크기
#define INPUT_SIZE 4096       // 픽셀 데이터 크기

// 데이터 구조 정의
typedef struct {
    float** pixel_data;  // 픽셀 데이터 배열
    int** labels;        // 원-핫 인코딩 레이블 배열
    int size;            // 데이터 크기 (행 수)
} Dataset;

// 정규화 함수
float normalize_pixel(char* value) {
    return atof(value) / 255.0f;
}

// 데이터 로드 함수
Dataset load_data(const char* base_dir, const char* sub_dir, const char* file_name) {
    // 파일 경로 생성
    char file_path[512];
    snprintf(file_path, sizeof(file_path), "%s//%s//%s", base_dir, sub_dir, file_name);

    // 파일 경로 출력
    printf("Trying to open file: %s\n", file_path);

    // 파일 열기
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    // 데이터 구조 초기화
    Dataset dataset;
    dataset.pixel_data = (float**)malloc(sizeof(float*) * 10000);  // 최대 10,000개 행 가정
    dataset.labels = (int**)malloc(sizeof(int*) * 10000);
    dataset.size = 0;

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        // CSV 한 줄 읽기
        char* token = strtok(line, ",");
        int label = atoi(token) - 1;  // 첫 번째 값: 레이블 (0~110로 변환)

        // 레이블 원-핫 인코딩
        int* one_hot_label = (int*)calloc(OUTPUT_SIZE, sizeof(int));
        if (label >= 0 && label < OUTPUT_SIZE) {
            one_hot_label[label] = 1;
        }
        else {
            fprintf(stderr, "Invalid label: %d\n", label);
            exit(1);
        }

        // 픽셀 데이터 생성
        float* pixels = (float*)malloc(sizeof(float) * INPUT_SIZE);
        int index = 0;

        while ((token = strtok(NULL, ",")) != NULL && index < INPUT_SIZE) {
            pixels[index++] = normalize_pixel(token);
        }

        // 데이터 저장
        dataset.pixel_data[dataset.size] = pixels;
        dataset.labels[dataset.size] = one_hot_label;
        dataset.size++;
    }

    fclose(file);
    return dataset;
}

// 동적 메모리 해제
void free_dataset(Dataset dataset) {
    for (int i = 0; i < dataset.size; i++) {
        free(dataset.pixel_data[i]);
        free(dataset.labels[i]);
    }
    free(dataset.pixel_data);
    free(dataset.labels);
}

// 메인 함수
int main() {
    const char* base_dir = "한글글자데이터";

    // Train 데이터 로드
    Dataset train_dataset = load_data(base_dir, "train", "train_data.csv");
    printf("Train Data Loaded: %d samples\n", train_dataset.size);

    // Test 데이터 로드
    Dataset test_dataset = load_data(base_dir, "test", "test_data.csv");
    printf("Test Data Loaded: %d samples\n", test_dataset.size);

    // 데이터 일부 확인
    for (int i = 0; i < 5 && i < train_dataset.size; i++) {
        printf("Train Label %d: ", i + 1);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (train_dataset.labels[i][j]) printf("%d ", j + 1);
        }
        printf("\nTrain Pixels: ");
        for (int j = 0; j < 10; j++) {  // 첫 10개 픽셀만 출력
            printf("%.3f ", train_dataset.pixel_data[i][j]);
        }
        printf("\n\n");
    }

    // 메모리 해제
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
