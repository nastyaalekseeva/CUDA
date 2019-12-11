
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;
#define BLOCK_SIZE 16
#define BASE_TYPE double
__global__ void matrixMult(const BASE_TYPE *A, BASE_TYPE *C, int Acols, int Arows)
{
	int i0 = Acols *(blockDim.y*blockIdx.y + threadIdx.y);
	//int  iAT = Arows*(blockDim.x*blockIdx.x + threadIdx.x) + blockDim.y*blockIdx.y + threadIdx.y;
	BASE_TYPE sum = 0;
	for (int k = 0; k < Acols; k++)
	{
		sum = +A[i0 + k] * A[i0+k];
	}
	int ind = Acols* (blockDim.y*blockIdx.y + threadIdx.y) + blockDim.x*blockIdx.x + threadIdx.x;
	C[ind] = sum;
}
int main()
{
	int Arows = 100;
	int Acols = 200;
	size_t Asize = Arows*Acols * sizeof(BASE_TYPE);
	BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
	BASE_TYPE *h_C = (BASE_TYPE *)malloc(Asize);
	for (int i = 0; i < Arows*Acols; i++)
	{
		h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < Arows*Acols; i++)
	{
		printf("h_A[%d]=%d ", i, h_A[i]);
	}
	BASE_TYPE *d_A = NULL;
	cudaMalloc((void **)&d_A, Asize);
	BASE_TYPE *d_C = NULL;
	cudaMalloc((void **)&d_C, Asize);
	cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(Acols / BLOCK_SIZE, Arows / BLOCK_SIZE);
	matrixMult <<<blocksPerGrid, threadsPerBlock >>> (d_A, d_C, Acols, Arows);
	cudaMemcpy(h_C, d_C, Asize, cudaMemcpyDeviceToHost);
	printf("Test Started\n");
	bool t = false;
	for (int i = 0; i < Arows; i++)
	{
		for (int j = 0; j < Arows; j++)
		{
			if (h_C[i*Arows + j] !=1)
			{
				t = true;
				//fprintf(stderr, "Result verification failed at element [%d,%d]!\n", i, j);
				//printf("sum=%f,h_C[i*Arows + j]=%f\n", 1, h_C[i*Arows + j]);
				//exit(EXIT_FAILURE);
				printf("Matrix A is not orthogonal\n");
				
			}
			if (t) break;
		}
		if (t) break;
	}
	printf("Test Passed\n");
	cudaFree(d_A);
	cudaFree(d_C);
	free(h_A);
	free(h_C);
	getchar();
	system("pause");
}
