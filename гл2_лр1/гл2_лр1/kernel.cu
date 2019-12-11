

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
int  main()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device name:%s\n", deviceProp.name);
	printf("Total global memory:%u\n", deviceProp.totalGlobalMem);
	printf("Total constant memory:%d\n", deviceProp.totalConstMem);
	printf("Shared memory per block:%d\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block:%d\n", deviceProp.regsPerBlock);
	printf("Warp size:%d\n", deviceProp.warpSize);
	printf("Max threads per block:%d\n", deviceProp.maxThreadsPerBlock);
	printf("Computer capabiliti:%d\n", deviceProp.major, deviceProp.minor);
	printf("Multiprocessor count:%d\n", deviceProp.multiProcessorCount);
	printf("Clock rate:%d\n", deviceProp.clockRate);
	printf("Memory clock rate:%d\n", deviceProp.memoryClockRate);
	printf("L2 cache:%d\n", deviceProp.l2CacheSize);
	printf("Memory bus width:%d\n", deviceProp.memoryBusWidth);
	printf("Max threads dimensions:x=%d,y=%d,z=%d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

	system("pause");

}