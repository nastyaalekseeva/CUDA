
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define N 100
#define BLOCK_SIZE 10
#define BASE_TYPE float

__global__ void scalMult(const BASE_TYPE *A, BASE_TYPE *C, int n)
{
	// ���������� ��� �������� ����� ���������  
	BASE_TYPE sum = 0.0;  // �������� �������� � ����������� ������  
	__shared__ BASE_TYPE ash[BLOCK_SIZE];
	// ����������� �� ���������� ������ 
	ash[threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * A[blockIdx.x * blockDim.x + threadIdx.x];
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
			sum = sum + ash[j];
		atomicAdd(C, sum);
		*C = __fsqrt_rz(*C);
	}

}
int main()
{
	BASE_TYPE *dev_a,  *dev_c;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// ���������� ��������     

	// ��������� ������ ��� ������� �� GPU   
	size_t size = N * sizeof(BASE_TYPE);
	BASE_TYPE *host_a = (BASE_TYPE *)malloc(size);
	BASE_TYPE *host_c = (BASE_TYPE *)malloc(size / N);
	for (int i = 0; i<N; i++)
	{
		host_a[i] = 0.2;
	}
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_c, size / N);
	// ����������� ������ � ������ GPU    
	cudaMemcpy(dev_a, host_a, size, cudaMemcpyHostToDevice);
	// ����� ����   
	cudaEventRecord(start, 0);
	scalMult << <BLOCK_SIZE, N / BLOCK_SIZE >> >(dev_a, dev_c, N);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// ���������� ������� ������ 
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %.2f milliseconds\n", KernelTime);
	cudaMemcpy(host_c, dev_c, size / N, cudaMemcpyDeviceToHost);
	// ����� �����������
	BASE_TYPE S = 0.0;
	for (int i = 0; i < N; i++)
	{
		S = S + host_a[i] * host_a[i];
	}
	S = sqrt(S);
	printf("<a,b>GPU = %f\n", *host_c);
	printf("<a,b> CPU= %f\n", S);
	// ������������ ������ 
	cudaFree(dev_a);
	cudaFree(dev_c);
	getchar();
	return 0;
}