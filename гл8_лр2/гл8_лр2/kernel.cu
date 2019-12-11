#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <assert.h>
#define N 10000
void checkCUDAError(const char *msg) 
{ 
	cudaError_t err = cudaGetLastError();  
	if (cudaSuccess != err) 
	{ 
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));   
		exit(EXIT_FAILURE); 
	} 
}
texture<float, 1, cudaReadModeElementType> texRef;
__global__ void Integral( float *C, float h, int n)
{
		int j = blockIdx.x*blockDim.x + threadIdx.x;

		if (j < n)

		{
			float x = ((tex1Dfetch(texRef, j)+ (tex1Dfetch(texRef, j) + *h))) / 2;
			C[j]= h*sqrt(1 - x*x);
		}

}
#define  NUM_THREADS 100 
int main()
{
	int nBlocks = N/NUM_THREADS;
	float a = 0, b = 1;
	float h = ((b - a)*1.0) / N;
	float x[N], S[N];//CPU
	float *dev_S; float *dev_x, *dev_h;//GPU
	x[0] = a;
	for (int i = 1; i < N; i++)
	{
		x[i] = i*h;
	}
	int size = sizeof(float);
	int memSize = N * sizeof(float);
	cudaMalloc((void**)&dev_S, memSize);
	cudaMalloc((void**)&dev_x, memSize);
	cudaMalloc((void**)&dev_h, size);
	cudaMemcpy(dev_x, &x, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, &h, size, cudaMemcpyHostToDevice);
	cudaBindTexture(0, texRef, dev_x,memSize);
	checkCUDAError("bind");
	Integral << < nBlocks, NUM_THREADS >> >(dev_S, dev_h,N);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(&S, dev_S,N*size, cudaMemcpyDeviceToHost);
	float p = 0.0;
	for (int i = 0; i < N; i++)
	{
		p = p + S[i];
	}
	cudaFree(dev_S);
	cudaUnbindTexture(texRef);
	printf("Pi = %f ", 4 * p);
	getchar();
}
