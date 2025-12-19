#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define M 69
#define N 69
#define O 69

#define BLOCK_SIZE 52


__global__ void matmul_gpu(float *A, float *B, float * C, int m, int n, int o){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col< o){
        float sum = 0.0f;
        for( int l = 0;l<n; l++){
            sum+=A[row*n + l] * B[l * o +col];
        }
        C[row*o + col] = sum;
    }
}

void init_matrix(float *mat, int rows, int cols){
    for (int i = 0; i<rows*cols; i++){
        mat[i] = (float)rand()/RAND_MAX;
    }
}

int main(){
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M*N*sizeof(float);
    int size_B = N*O*sizeof(float);
    int size_C = O*M*sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);
    
    init_matrix(h_A, M, N);
    init_matrix(h_B, N, O);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((O + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Starting GPU Matrix Multiplication\n");
    matmul_gpu<<<gridDim, blockDim>>> (d_A, d_B, d_C, M, N, O);
    cudaDeviceSynchronize();

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_B);

    return 0;
}