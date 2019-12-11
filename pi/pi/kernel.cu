
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#define  n 10000
#define BLOCK 1000
 __global__ void Su(float *a,float *b,float *h)
 {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	b[i] = (*h)*sqrtf(1 - a[i] * a[i]);
 }
int main()
{
	float a = 0, b = 1;
	float h = ((b - a)*1.0) / n;
	float x[n],S[n];//CPU
	float *dev_S,*dev_x,*dev_h;//GPU
	x[0] = a;
	for (int i = 1; i < n; i++)
	{
		x[i] = i*h;
	}
	int size = sizeof(float);
	cudaMalloc((void**)&dev_S, n*size);
	cudaMalloc((void**)&dev_x, n*size);
	cudaMalloc((void**)&dev_h, size);
	cudaMemcpy(dev_x, &x,n*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, &h, size, cudaMemcpyHostToDevice);
	Su<<< BLOCK,n/BLOCK >>>(dev_x,dev_S,dev_h);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf( "GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(&S, dev_S, n*size, cudaMemcpyDeviceToHost);
	float p = 0.0;
	for (int i = 0; i < n; i++)
	{
		p=p+S[i];
	}
	cudaFree(dev_S);
	printf("Pi = %f ", 4*p);
	getchar();
}
