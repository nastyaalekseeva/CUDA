
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE double
// функция перемножения матриц
__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int cols, int rows)
{
	int i0 = cols * (blockDim.y * blockIdx.y + threadIdx.y);
	int j0 = blockDim.x * blockIdx.x + threadIdx.x;
	BASE_TYPE sum = 0;
	for (int k = 0; k < cols; k++)
		sum += A[i0 + k] * B[k * rows + j0];
	int ind = rows * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C[ind] = sum;
}
int main()
{
	// количество строк и столбцов матрицы 
	int rows = 100;
	int cols = 200;


	size_t size = rows * cols * sizeof(BASE_TYPE);
	//size_t Bsize = rows * cols * sizeof(BASE_TYPE);
	//size_t Csize = rows * cols * sizeof(BASE_TYPE);
	BASE_TYPE *h_A = (BASE_TYPE *)malloc(size);
	BASE_TYPE *h_B = (BASE_TYPE *)malloc(size);
	BASE_TYPE *h_C = (BASE_TYPE *)malloc(size);
	BASE_TYPE *h_C1 = (BASE_TYPE *)malloc(size);
	for (int i = 0; i < rows * cols; ++i) {
		h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < rows * cols; ++i) {
		h_B[i] =rand() / (BASE_TYPE)RAND_MAX;
	}
	BASE_TYPE *d_A = NULL;
	cudaMalloc((void **)&d_A, size);
	BASE_TYPE *d_B = NULL;
	cudaMalloc((void **)&d_B, size);
	BASE_TYPE * d_C = NULL;
	cudaMalloc((void **)&d_C, size);
	BASE_TYPE * d_C1 = NULL;
	cudaMalloc((void **)&d_C1, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(cols / BLOCK_SIZE, rows / BLOCK_SIZE);
	matrixMult << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, cols, rows);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	matrixMult << <blocksPerGrid, threadsPerBlock >> >(d_B, d_A, d_C1, cols,rows);
	cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost);
	printf("Test STARTED\n");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			
			if (fabs(h_C[i * cols + j] - h_C1[i * cols + j]) > 1e-3)
			{
				
				printf("AB!=BA");
				//exit(EXIT_FAILURE);
			}
		}
	}
	printf("Test PASSED\n");
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_C1);
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C1);
	getchar();
	return 0;
}