
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#define  n 100000
#define BLOCK 1000
__global__ void Center(float *x, float *C, float *h)
{
	float sum = 0.0;
	__shared__ float xx[n / BLOCK];
	xx[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
	// Синхронизация нитей  
	__syncthreads();
	// Вычисление скалярного произведения  
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		
		//центральные прямоугольники
	
		for (int j = 0; j < blockDim.x; j++)
		{
			float xh2 = xx[j] - *h / 2;
			sum += (*h)*sqrt(1-xh2*xh2);
		}

		C[blockIdx.x] = sum;
	}
}
__global__ void Trap(float *x, float *C, float *h)
{
	float sum = 0.0;
	__shared__ float xx[n / BLOCK];
	xx[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
	// Синхронизация нитей  
	__syncthreads();
	// Вычисление скалярного произведения  
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		// трапеция
		for (int j = 1; j < blockDim.x; j++)
		{
		sum += (*h) * (sqrt(1-xx[j] * xx[j])+sqrt(1-xx[j - 1] * xx[j - 1]))/2;
		}
		C[blockIdx.x] = sum;
	}
}
__global__ void Simc(float *x, float *C, float *h)
{
	float sum = 0.0;
	__shared__ float xx[n / BLOCK];
	xx[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
	// Синхронизация нитей  
	__syncthreads();
	// Вычисление скалярного произведения  
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		//Симпсона
		for (int j = 0; j < blockDim.x; j=j+2)
		{
		sum += (*h) / 3 *( sqrt(1-xx[j] * xx[j])+4*sqrt(1-xx[j + 1] * xx[j + 1])+ sqrt(1-xx[j + 2] * xx[j + 2]));
		}

		C[blockIdx.x] = sum;
	}
}
int main()
{
	float a = 0, b = 1;
	float h = ((b - a)*1.0) / n;
	float x[n], S[n];//CPU
	float *dev_S, *dev_x, *dev_h;//GPU
	x[0] = a;
	for (int i = 1; i < n; i++)
	{
		x[i] = i*h;
	}
	int size = sizeof(float);

	cudaMalloc((void**)&dev_S, n*size);
	cudaMalloc((void**)&dev_x, n*size);
	cudaMalloc((void**)&dev_h, size);
	cudaMemcpy(dev_x, &x, n*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, &h, size, cudaMemcpyHostToDevice);
	Center << < BLOCK, n / BLOCK >> >(dev_x, dev_S, dev_h);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(&S, dev_S, n*size, cudaMemcpyDeviceToHost);
	float p = 0.0;
	for (int i = 0; i < n; i++)
	{
		p = p + S[i];
	}
	//cudaFree(dev_S);
	printf("Pi Center = %f \n", 4*p);

	Trap << < BLOCK, n / BLOCK >> >(dev_x, dev_S, dev_h);
	cudaDeviceSynchronize();
	//cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(&S, dev_S, n*size, cudaMemcpyDeviceToHost);
	 p = 0.0;
	for (int i = 0; i < n; i++)
	{
		p = p + S[i];
	}
	//cudaFree(dev_S);
	printf("Pi Trap= %f \n", 4 * p);
	Simc<< < BLOCK, n / BLOCK >> >(dev_x, dev_S, dev_h);
	cudaDeviceSynchronize();
	//cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(&S, dev_S, n*size, cudaMemcpyDeviceToHost);
	 p = 0.0;
	for (int i = 0; i < n; i++)
	{
		p = p + S[i];
	}
	cudaFree(dev_S);
	printf("Pi Sim = %f ", 4 * p);
	getchar();
}
