
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#define  n 100000
#define BLOCK 1000
__global__ void MK(int *S, unsigned int seed)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	float x, y;
	curandState_t m;
	curand_init(seed, i, 0, &m);
	x = curand_uniform(&m);
	curand_init(NULL, i, 0, &m);
	y =curand_uniform(&m);
	//printf("x=%f, y=%f \n", x,y);
	if (x*x + y*y <= 1)
		atomicAdd(S, 1);
	
}
void main()
{
	int host_S;
	int *dev_S;
	int size = sizeof(int);
	cudaMalloc((void**)&dev_S, size);
	MK <<< BLOCK,n/BLOCK >>>(dev_S, time(NULL));
	cudaDeviceSynchronize();
	cudaMemcpy(&host_S, dev_S, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_S);
	printf("pi=%f", (4 * host_S*1.0)/n);

	system("pause");
}
