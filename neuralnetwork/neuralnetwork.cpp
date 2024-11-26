#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 4096  // CSV ���� �� ���� �ִ� ����
#define OUTPUT_SIZE 111       // ��-�� ���ڵ� ũ��
#define INPUT_SIZE 4096       // �ȼ� ������ ũ��

// ������ ���� ����
typedef struct {
    float** pixel_data;  // �ȼ� ������ �迭
    int** labels;        // ��-�� ���ڵ� ���̺� �迭
    int size;            // ������ ũ�� (�� ��)
} Dataset;

// ����ȭ �Լ�
float normalize_pixel(char* value) {
    return atof(value) / 255.0f;
}

// ������ �ε� �Լ�
Dataset load_data(const char* base_dir, const char* sub_dir, const char* file_name) {
    // ���� ��� ����
    char file_path[512];
    snprintf(file_path, sizeof(file_path), "%s//%s//%s", base_dir, sub_dir, file_name);

    // ���� ��� ���
    printf("Trying to open file: %s\n", file_path);

    // ���� ����
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    // ������ ���� �ʱ�ȭ
    Dataset dataset;
    dataset.pixel_data = (float**)malloc(sizeof(float*) * 10000);  // �ִ� 10,000�� �� ����
    dataset.labels = (int**)malloc(sizeof(int*) * 10000);
    dataset.size = 0;

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        // CSV �� �� �б�
        char* token = strtok(line, ",");
        int label = atoi(token) - 1;  // ù ��° ��: ���̺� (0~110�� ��ȯ)

        // ���̺� ��-�� ���ڵ�
        int* one_hot_label = (int*)calloc(OUTPUT_SIZE, sizeof(int));
        if (label >= 0 && label < OUTPUT_SIZE) {
            one_hot_label[label] = 1;
        }
        else {
            fprintf(stderr, "Invalid label: %d\n", label);
            exit(1);
        }

        // �ȼ� ������ ����
        float* pixels = (float*)malloc(sizeof(float) * INPUT_SIZE);
        int index = 0;

        while ((token = strtok(NULL, ",")) != NULL && index < INPUT_SIZE) {
            pixels[index++] = normalize_pixel(token);
        }

        // ������ ����
        dataset.pixel_data[dataset.size] = pixels;
        dataset.labels[dataset.size] = one_hot_label;
        dataset.size++;
    }

    fclose(file);
    return dataset;
}

// ���� �޸� ����
void free_dataset(Dataset dataset) {
    for (int i = 0; i < dataset.size; i++) {
        free(dataset.pixel_data[i]);
        free(dataset.labels[i]);
    }
    free(dataset.pixel_data);
    free(dataset.labels);
}

// ���� �Լ�
int main() {
    const char* base_dir = "�ѱ۱��ڵ�����";

    // Train ������ �ε�
    Dataset train_dataset = load_data(base_dir, "train", "train_data.csv");
    printf("Train Data Loaded: %d samples\n", train_dataset.size);

    // Test ������ �ε�
    Dataset test_dataset = load_data(base_dir, "test", "test_data.csv");
    printf("Test Data Loaded: %d samples\n", test_dataset.size);

    // ������ �Ϻ� Ȯ��
    for (int i = 0; i < 5 && i < train_dataset.size; i++) {
        printf("Train Label %d: ", i + 1);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (train_dataset.labels[i][j]) printf("%d ", j + 1);
        }
        printf("\nTrain Pixels: ");
        for (int j = 0; j < 10; j++) {  // ù 10�� �ȼ��� ���
            printf("%.3f ", train_dataset.pixel_data[i][j]);
        }
        printf("\n\n");
    }

    // �޸� ����
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
