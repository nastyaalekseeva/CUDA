
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define N 128 
#define BLOCK_SIZE 16
#define BASE_TYPE float

__global__ void scalMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C ,int n)
{ 
	// Переменная для хранения суммы элементов  
    BASE_TYPE sum = 0.0;  // Создание массивов в разделяемой памяти  
	__shared__ BASE_TYPE ash[BLOCK_SIZE];
	// Копирование из глобальной памяти 
	ash[threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x]*B[blockIdx.x * blockDim.x + threadIdx.x]; 
	if(threadIdx.x==0)
	{
		sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
			sum = sum+ash[j];
		atomicAdd(C, sum);
	}
	
}
int main() 
{         
	BASE_TYPE *dev_a, *dev_b, *dev_c;
	cudaEvent_t start, stop; 
	cudaEventCreate(&start);  
	cudaEventCreate(&stop);
	// заполнение массивов     
	    
	// выделение памяти под массивы на GPU   
	size_t size = N* sizeof(BASE_TYPE);
	BASE_TYPE *host_a = (BASE_TYPE *)malloc(size);
	BASE_TYPE *host_b = (BASE_TYPE *)malloc(size);
	BASE_TYPE *host_c = (BASE_TYPE *)malloc(size/N);
	for (int i = 0; i<N; i++)
	{
		host_a[i] = 1;
		host_b[i] = 5;
	}
	cudaMalloc( (void**)&dev_a, size );
	cudaMalloc( (void**)&dev_b, size);
	cudaMalloc( (void**)&dev_c, size/N);
				 // копирование данных в память GPU    
	cudaMemcpy( dev_a, host_a, size, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( dev_b, host_b, size, cudaMemcpyHostToDevice ) ;
				 // вызов ядра   
	cudaEventRecord(start, 0);
	scalMult <<<BLOCK_SIZE, N / BLOCK_SIZE>>>( dev_a, dev_b, dev_c,N);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	// вычисление времени работы 
	float KernelTime; 
	cudaEventElapsedTime( &KernelTime, start, stop); 
	printf("KernelTime: %.2f milliseconds\n", KernelTime);
	cudaMemcpy(host_c, dev_c, size/N, cudaMemcpyDeviceToHost);
	// вывод результатов
	float S = 0.0;
	for (int i = 0; i < N; i++)
	{
		S = S + host_a[i]*host_b[i];
	}
	printf( "<a,b>GPU = %f\n", *host_c); 
	printf("<a,b> CPU= %f\n", S);
	// освобождение памяти 
	cudaFree( dev_a ); 
	cudaFree( dev_b ); 
	cudaFree( dev_c ); 
	getchar();
	return 0; 
}