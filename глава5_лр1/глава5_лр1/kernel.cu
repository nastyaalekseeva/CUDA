
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#define n 1000
#define MAX_GRIDSIZE 1
__device__ float __expf(float x);
__global__ void expMass(float *A,float *x, int arraySize)
{
	int index = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (index < arraySize)
		A[index]=expf(x[index]);
}
int main()
{
	float a = 1, b = 5;
	float h = ((10-1)*1.0)/n;
	float x[n], S[n],arr[n];
	float *A, *dev_x;
	x[0] = a;
	for (int i = 1; i < n; i++)
	{
		x[i] = i*h;
		arr[i] = exp(x[i]);
	}

	int size = sizeof(float);
	cudaMalloc((void**)&A, n*size);
	cudaMalloc((void**)&dev_x, n*size);
	cudaMemcpy(dev_x, &x, n*size, cudaMemcpyHostToDevice);
	expMass << <1, n >> > (A, dev_x,n);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(S, A, n*size, cudaMemcpyDeviceToHost);
	for (int i = 1; i < n; i++)
	{
		printf("x=%f,  S=%f,  arr=%f,  err=%f \n", x[i], S[i], arr[i],abs(S[i] - arr[i]) / n);
	}
	getchar();
}