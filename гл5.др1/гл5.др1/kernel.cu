
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
__global__ void a_f(float *A, float *x,float *y)
{
		*A = __fadd_rz(*x,*y);
}

__global__ void mul_f(float *A, float *x, float *y)
{
		*A = __fmul_rz(*x, *y);
}
__global__ void sqrt_f(float *A, float *x)
{
		*A = __fsqrt_rz(*x);
}
__global__ void a_d(double *A, double *x, double *y)
{
		*A = __dadd_rz(*x, *y);
}

__global__ void mul_d(double *A, double *x, double *y)
{
		*A = __dmul_rz(*x, *y);
}
__global__ void sqrt_d(double *A, double *x)
{
		*A = __dsqrt_rz(*x);
}
int main()
{
	float x=5.1553250, S[1], y=4.5467960;
	float *A, *dev_x, *dev_y;
	double arr[1],xd = 5.155325, yd = 4.546796;
	double *A_d, *dev_xd, *dev_yd;

	int size = sizeof(float);
	cudaMalloc((void**)&A, size);
	cudaMalloc((void**)&dev_x, size);
	cudaMemcpy(dev_x, &x, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_y, size);
	cudaMemcpy(dev_y, &y, size, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop; 
	float elapsedTime;
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);  
	cudaEventRecord(start, 0); 
	a_f << <1, 1 >> > (A, dev_x,dev_y);
	//mul_f << <1, 1 >> > (A, dev_x, dev_y);
	//sqrt_f << <1, 1 >> > (A, dev_x);
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err));
	}
	cudaMemcpy(S, A, size, cudaMemcpyDeviceToHost);

	int size_d = sizeof(double);
	cudaMalloc((void**)&A_d, size_d);
	cudaMalloc((void**)&dev_xd, size_d);
	cudaMemcpy(dev_xd, &xd, size_d, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_yd, size_d);
	cudaMemcpy(dev_yd, &yd, size_d, cudaMemcpyHostToDevice);
	cudaEvent_t start1, stop1;
	float elapsedTime1; 
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);  
	cudaEventRecord(start1, 0);
	a_d << <1, 1 >> > (A_d, dev_xd, dev_yd);
	//mul_d << <1, 1 >> > (A_d, dev_xd, dev_yd);
	//sqrt_d << <1, 1 >> > (A_d, dev_xd);
	cudaEventRecord(stop1, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop1); 
	cudaEventElapsedTime(&elapsedTime1, start1, stop1); 
	printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime1); 
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cudaError_t err1 = cudaGetLastError();
	if (err1 != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(err1));
	}
	cudaMemcpy(arr, A_d, size_d, cudaMemcpyDeviceToHost);

		printf("x=%f, y=%f,  S=%f,  arr=%f \n", x,y, S[0], arr[0]);

	getchar();
}