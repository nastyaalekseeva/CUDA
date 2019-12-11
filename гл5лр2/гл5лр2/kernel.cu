
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#define N 1000
#define BLOCK_SIZE 10
 __global__ void scaMult_g (int *a, int *b, int *c, int *sum, int n)
{

	int tid = threadIdx.x;
	if (tid > n- 1) return;
	{
		c[tid] = a[tid] * b[tid];
		atomicAdd(sum, c[tid]);
	}
}
int main() {

	int *host_a = new int[N];
	int *host_b = new int[N];
	int *a = new int[N];
	int *b = new int[N];
	int *sum = 0;
	int  *dev_c, *dev_sum, *dev_a, *dev_b, host_sum;
	for (int i = 0; i < N; i++)
	{
		host_a[i] = 8;
		host_b[i] = 1;

	}
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	cudaMalloc((void**)&dev_sum, sizeof(int));
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sum, sum, sizeof(int), cudaMemcpyHostToDevice);
	scaMult_g <<< BLOCK_SIZE, N/BLOCK_SIZE >>> (dev_a, dev_b, dev_c, dev_sum, N);
	cudaMemcpy(&host_sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost);
	int s = 0;
	for (int i = 0; i < N; i++)
	{
		s = s + host_a[i] * host_b[i];

	}
	printf("g <a,b>=%d \n", host_sum);
	printf("c <a,b>=%d \n", s);
	getchar();
}