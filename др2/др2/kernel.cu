
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>
#define BLOCK_SIZE 10
#define THREADS_PER_BLOCK 10
#define N 10
#define M 10
#define BASE_TYPE float
__global__ void vectorSub(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *c, int cols)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < cols)
		c[tid] = a[tid] + b[tid];
}

__global__ void show(BASE_TYPE **A, int cols, int rows)
{
	printf("Matrix on GPU:\n");
	for (int i = 0; i != rows; i++)
	{
		for (int j = 0; j != cols; j++)
		{
			printf("%f  ", A[i][j]);
		}
		printf("\n");
	}
}
int main()
{
	int cols = N;
	int rows = M;


	BASE_TYPE **h_A = (BASE_TYPE **)malloc(rows * sizeof(BASE_TYPE*));
	for (int i = 0; i < rows; i++) {
		h_A[i] = (BASE_TYPE*)malloc(cols * sizeof(BASE_TYPE));
	}
	BASE_TYPE **h_B = (BASE_TYPE **)malloc(rows * sizeof(BASE_TYPE*));
	for (int i = 0; i < rows; i++) {
		h_B[i] = (BASE_TYPE*)malloc(cols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; j++)
		{
			h_A[i][j] = 2;
		}
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; j++)
			h_B[i][j] = 5;
	}

	BASE_TYPE **h_C = (BASE_TYPE **)malloc(rows * sizeof(BASE_TYPE*));
	for (int i = 0; i < rows; i++) {
		h_C[i] = (BASE_TYPE*)malloc(cols * sizeof(BASE_TYPE));
	}

	BASE_TYPE **d_A = NULL;
	cudaMalloc((void **)&d_A, rows * sizeof(BASE_TYPE*));
	BASE_TYPE **h_a = (BASE_TYPE **)malloc(rows * sizeof(BASE_TYPE*));
	for (int i = 0; i<rows; i++) {
		cudaMalloc((void**)&h_a[i], cols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<rows; i++) {
		cudaMemcpy(h_a[i], h_A[i], cols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_A, h_a, rows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);


	BASE_TYPE **d_B = NULL;
	cudaMalloc((void **)&d_B, rows * sizeof(BASE_TYPE*));
	BASE_TYPE **h_b = (BASE_TYPE **)malloc(rows * sizeof(BASE_TYPE*));
	for (int i = 0; i<rows; i++) {
		cudaMalloc((void**)&h_b[i], cols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<rows; i++) {
		cudaMemcpy(h_b[i], h_B[i], cols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_B, h_b, rows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);

	BASE_TYPE **d_C = NULL;
	cudaMalloc((void **)&d_C, rows * sizeof(BASE_TYPE*));
	BASE_TYPE **h_c = (BASE_TYPE **)malloc(rows * sizeof(BASE_TYPE*));
	for (int i = 0; i<rows; i++) {
		cudaMalloc((void**)&h_c[i], cols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<rows; i++) {
		cudaMemcpy(h_c[i], h_C[i], cols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_C, h_c, rows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);

	for (int i = 0; i < rows; i++)
	{
		vectorSub << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (h_a[i], h_b[i], h_c[i], cols);

	}
	printf("A+B=C matrix:\n");
	show << <1, 1 >> >(d_C, cols, rows);
	cudaDeviceSynchronize();
	getchar();
}