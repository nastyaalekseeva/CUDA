#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
//����
__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}
//������� �������
int main()
{
	
	int a, b, c; // ���������� �� CPU
	int *dev_a, *dev_b, *dev_c; // ���������� �� GPU
	int size = sizeof(int); //�����������
							//�������� ������ �� GPU
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	//������������� ����������
	a = 2;
	b = 7;
	// ����������� ���������� � CPU �� GPU
	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;//�������������� �������
	float elapsedTime;
	cudaEventCreate(&start);//������� �������
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);//������ ������� 
	// ����� ����
	add <<< 1, 1 >>>(dev_a, dev_b, dev_c);
	//����������� ���������� ������ ���� � GPU �� CPU
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	//�������� ������ �� GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//�������� ���������� ������ ����
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime
		);//����� ����������
	cudaEventDestroy(start);//����������� �������
	cudaEventDestroy(stop);
	system("pause");
}