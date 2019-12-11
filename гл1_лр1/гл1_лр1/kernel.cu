
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
using namespace std;
int  main()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device name:%s\n", deviceProp.name);
	printf("Multiprocessor count:%d\n", deviceProp.multiProcessorCount);
	printf("Total global memory:%u\n", deviceProp.totalGlobalMem);
	printf("Memory clock rate:%d\n", deviceProp.memoryClockRate);
	printf("Clock rate:%d\n", deviceProp.clockRate); 
	printf("Memory bus width:%d\n", deviceProp.memoryBusWidth);
	scanf("");
	system("pause");
}