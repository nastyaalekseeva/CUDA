
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define N 128 
#define BLOCK_SIZE 16
#define BASE_TYPE float
__device__ __constant__ BASE_TYPE ash[N];
__device__ __constant__ BASE_TYPE bsh[N];
__global__ void scalMult( BASE_TYPE *C)
{
	BASE_TYPE sum = 0.0;
	__syncthreads();  
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
			sum += ash[j] * bsh[j];
		C[blockIdx.x] = sum;
	}
}
int main()
{
	BASE_TYPE *dev_a, *dev_b, *dev_c;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);     
	size_t size = N * sizeof(BASE_TYPE);
	BASE_TYPE *host_a = (BASE_TYPE *)malloc(size);
	BASE_TYPE *host_b = (BASE_TYPE *)malloc(size);
	BASE_TYPE *host_c = (BASE_TYPE *)malloc(size);
	for (int i = 0; i<N; i++)
	{
		host_a[i] = 5;
		host_b[i] = 5;
	}
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);    
	cudaMemcpyToSymbol(ash, host_a, size,0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(bsh, host_b, size,0, cudaMemcpyHostToDevice);  
	cudaEventRecord(start, 0);
	scalMult << <BLOCK_SIZE, N / BLOCK_SIZE >> >(dev_c);
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %.2f milliseconds\n", KernelTime);
	cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);  
	BASE_TYPE S = 0;
	for (int i = 0; i < N; i++)
	{
		S = S + host_c[i];
	}
	printf("<a,b>on GPU = %f\n", S);
	float S1 = 0.0;
	for (int i = 0; i < N; i++)
	{
		S1 += host_a[i]*host_b[i];
	}
	printf("<a,b> on CPU = %f\n", S1);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	getchar();
	return 0;
}
