
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#define  n 10000
#define BLOCK 10
__global__ void Su(float *S_d, float *x)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	float q = 1.0;
	for (int j = 1; j <= *x; j++)
	{
		q = q*i;
	}
	S_d[i] =  1./q;	
}
int main()
{

	float host_x=2;//CPU
	float *dev_S;float *dev_x;//GPU
	float host_S[n];
	int size = sizeof(float);
	cudaMalloc((void**)&dev_S, n*size);
	cudaMalloc((void**)&dev_x, size);
	cudaMemcpy(dev_S, &host_S, n*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, &host_x, size, cudaMemcpyHostToDevice);
	
	Su <<< BLOCK, n / BLOCK >>>(dev_S,dev_x);
	cudaDeviceSynchronize();
	cudaMemcpy(&host_S, dev_S, n*size, cudaMemcpyDeviceToHost);
	float p = 0.0;
	for (int i = 1; i < n; i++)
	{
		p = p + host_S[i];
	}
	printf("S = %f ", p);
	cudaFree(dev_S);
	cudaFree(dev_x);
	
	getchar();
}
