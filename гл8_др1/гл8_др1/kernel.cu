
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <assert.h>
#define  NUM_THREADS 100
//#define N 10000
texture<float, 1, cudaReadModeElementType> texRef1;
texture<float, 1, cudaReadModeElementType> texRef2;
texture<float, 1, cudaReadModeElementType> texRef3;
texture<float, 1, cudaReadModeElementType> texRef4;
__global__ void scalMult(float *C, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		float x = tex1D(texRef1, float(idx));
		float y= tex1D(texRef2, float(idx));
		C[idx] = x*y;
	}
}

__global__ void scalMult1(float *C, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		float y = tex1Dfetch(texRef3, idx);
		float x = tex1D(texRef1, float(idx));
		//printf("j=%d x=%f, y=%f \n", idx, x, y);
		C[idx] = x*y;
	}
}
__global__ void scalMult2(float *C, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		float y = tex1Dfetch(texRef3, idx);
		float x = tex1Dfetch(texRef4, idx);
		//printf("j=%d x=%f, y=%f \n", idx, x, y);
		C[idx] = x*y;
	}
}
void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
int main()
{
	float *dev_S;
	float *h_S;
	for (int N = 1000; N <= 50000; N = N + 1000)
	{
		int nBlocks = N / NUM_THREADS;
		printf("N=%d\n",N);
		cudaMalloc((void**)&dev_S, sizeof(float)*N);
		h_S= (float*)malloc(sizeof(float)*N);
		float *x = (float*)malloc(N * sizeof(float));
		float *y = (float*)malloc(N * sizeof(float));
		for (int i = 0; i < N; i++)
		{
			x[i] = 2;
			y[i] = 1;
		}
		cudaArray* cuArray_x;
		cudaMallocArray(&cuArray_x, &texRef1.channelDesc, N, 1);
		cudaMemcpyToArray(cuArray_x, 0, 0, x, sizeof(float)*N, cudaMemcpyHostToDevice);
		cudaBindTextureToArray(texRef1, cuArray_x);
		cudaArray* cuArray_y;
		cudaMallocArray(&cuArray_y, &texRef2.channelDesc, N, 1);
		cudaMemcpyToArray(cuArray_y, 0, 0, y, sizeof(float)*N, cudaMemcpyHostToDevice);
		cudaBindTextureToArray(texRef2, cuArray_y);
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		scalMult << <nBlocks, NUM_THREADS >> > (dev_S, N);
		cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
		float KernelTime;
		cudaEventElapsedTime(&KernelTime, start, stop);
		printf("1. CUDA Array\n");
		printf("KernelTime: %.2f milliseconds\n", KernelTime);
		cudaMemcpy(h_S, dev_S, sizeof(float)*N, cudaMemcpyDeviceToHost);
		float sum = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum = sum + h_S[i];
		}
		printf("GPU: %f \n", sum);
		float sum1 = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum1 = sum1 + x[i] * y[i];
		}
		printf("CPU: %f \n", sum1);

		float *d_y1;
		int memSize = N * sizeof(float);
		cudaMalloc((void**)&d_y1, memSize);
		cudaMemcpy(d_y1, y, memSize, cudaMemcpyHostToDevice);
		cudaBindTexture(0, texRef3, d_y1, memSize);
		checkCUDAError("bind");
		cudaEvent_t start1, stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1, 0);
		scalMult1 << <nBlocks, NUM_THREADS >> > (dev_S, N);
		cudaDeviceSynchronize();
		cudaEventRecord(stop1, 0); 
		cudaEventSynchronize(stop1);
		float KernelTime1;
		cudaEventElapsedTime(&KernelTime1, start1, stop1);
		printf("2. CUDA Array and linear memory \n");
		printf("KernelTime: %.2f milliseconds\n", KernelTime1);
		cudaMemcpy(h_S, dev_S, memSize, cudaMemcpyDeviceToHost);
		sum = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum = sum + h_S[i];
		}
		printf("GPU: %f \n", sum);
		sum1 = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum1 = sum1 + x[i] * y[i];
		}
		printf("CPU: %f \n", sum1);


		cudaMalloc((void**)&d_y1, memSize);
		cudaMemcpy(d_y1, y, memSize, cudaMemcpyHostToDevice);
		cudaBindTexture(0, texRef3, d_y1, memSize);
		float *d_x1;
		cudaMalloc((void**)&d_x1, memSize);
		cudaMemcpy(d_x1, x, memSize, cudaMemcpyHostToDevice);
		cudaBindTexture(0, texRef4, d_x1, memSize);
		checkCUDAError("bind");
		cudaEvent_t start2, stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2, 0);
		scalMult2 << <nBlocks, NUM_THREADS >> > (dev_S, N);
		cudaDeviceSynchronize();
		cudaEventRecord(stop2, 0);
		cudaEventSynchronize(stop2);
		float KernelTime2;
		cudaEventElapsedTime(&KernelTime2, start2, stop2);
		printf("3. Linear memory \n");
		printf("KernelTime: %.2f milliseconds\n", KernelTime2);
		cudaMemcpy(h_S, dev_S, memSize, cudaMemcpyDeviceToHost);
		sum = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum = sum + h_S[i];
		}
		printf("GPU: %f \n", sum);
		sum1 = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum1 = sum1 + x[i] * y[i];
		}
		printf("CPU: %f \n", sum1);
		printf("\n");
	}
	free(h_S);
	cudaFree(dev_S);
	cudaUnbindTexture(texRef1);
	cudaUnbindTexture(texRef2);
	cudaUnbindTexture(texRef3);
	cudaUnbindTexture(texRef4);
	getchar();
}