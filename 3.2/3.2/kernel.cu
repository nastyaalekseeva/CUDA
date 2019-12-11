#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
//ядро
__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}
//главная функция
int main()
{
	
	int a, b, c; // переменные на CPU
	int *dev_a, *dev_b, *dev_c; // переменные на GPU
	int size = sizeof(int); //размерность
							//выделяем память на GPU
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	//инициализация переменных
	a = 2;
	b = 7;
	// Копирование информации с CPU на GPU
	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;//инициализируем события
	float elapsedTime;
	cudaEventCreate(&start);//создаем события
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);//запись события 
	// Вызов ядра
	add <<< 1, 1 >>>(dev_a, dev_b, dev_c);
	//Копирование результата работы ядра с GPU на CPU
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	//Очищение памяти на GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//ожидание завершения работы ядра
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime
		);//вывод информации
	cudaEventDestroy(start);//уничтожение события
	cudaEventDestroy(stop);
	system("pause");
}