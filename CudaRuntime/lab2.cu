#pragma once
#include "lab2.cuh"
#include "device_launch_parameters.h"

#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>

#define block_size 256

__host__ char* gen_text(int length);

__host__ void CPU_count_chars(char* text, int text_length, int* TAB);

__host__ void GPU_count_chars(char* text, int text_length, int* TAB);

__host__ bool compare_TABs(int* TAB1, int* TAB2);

__global__ void kernel_count_chars(char* text, int* TAB);

__host__ void lab2()
{
	int text_length;
	std::cout << "Text length: ";
	std::cin >> text_length;

	if (text_length % block_size)
		text_length += block_size - text_length % block_size;

	char* text = gen_text(text_length);

	int CPU_TAB[256]{ 0 }, GPU_TAB[256]{ 0 };

	auto cpu_beg = std::chrono::steady_clock().now();
	CPU_count_chars(text, text_length, CPU_TAB);
	auto cpu_end = std::chrono::steady_clock().now();
	std::cout << "cpu time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_beg).count() << std::endl;

	GPU_count_chars(text, text_length, GPU_TAB);

	cudaDeviceReset();

	/*int sum1 = 0, sum2 = 0;
	for (int i = 0; i < 256; sum1 += CPU_TAB[i], sum2 += GPU_TAB[i], i++);

	std::cout << "cpu count: " << sum1 << "\ngpu count: " << sum2 << std::endl;*/


	if (compare_TABs(CPU_TAB, GPU_TAB))
		std::cout << "TABs are same\n";
	else
		std::cout << "TABS aren't same\n";

	delete[] text;
}

__host__ void CPU_count_chars(char* text, int text_length, int* TAB)
{
#pragma omp parallel
{
		std::vector<int> _TAB(256,0);
#pragma omp for
		for (int i = 0; i < text_length; i++)
		{
#pragma omp atomic
			_TAB[text[i]]++;
		}
#pragma omp for
		for (int i = 0; i < 256; i++)
			TAB[i] = _TAB[i];
}
}

__host__ char* gen_text(int length)
{
	char* text = new char[length];
	for (int i = 0; i < length; text[i] = (rand() % 93) + 33, i++);
	return text;
}

__host__ bool compare_TABs(int* TAB1, int* TAB2)
{
	for (int i = 0; i < 256; i++)
		if (TAB1[i] != TAB2[i])
			return false;
	return true;
}

__host__ void GPU_count_chars(char* text, int text_length, int* TAB)
{
	char* d_text = nullptr;
	int* d_TAB = nullptr;

	cudaEvent_t start, stop;
	float elapsed_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_text, text_length * sizeof(char));
	cudaMalloc((void**)&d_TAB, 256 * sizeof(int));

	cudaMemcpy(d_text, text, text_length * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_TAB, TAB, 256 * sizeof(int), cudaMemcpyHostToDevice);

	int grid_size = ceil((double)text_length / (double)block_size);

	kernel_count_chars << <grid_size, block_size >> > (d_text, d_TAB);

	cudaDeviceSynchronize();

	cudaMemcpy(TAB, d_TAB, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_text);
	cudaFree(d_TAB);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "gpu_time: " << elapsed_time << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void kernel_count_chars(char* text, int* TAB)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	//atomicAdd(TAB + text[i], 1);

	__shared__ int tab[256];

	/*if (threadIdx.x == 0)
		for (int j = 0; j < 256; tab[j] = 0, j++);*/

	if (threadIdx.x < 256)
		tab[threadIdx.x] = 0;

	__syncthreads();
	
	atomicAdd(tab + text[i], 1);
	
	__syncthreads();
	
	/*if (threadIdx.x == 0)
		for (int j = 0; j < 256; atomicAdd(TAB + j, tab[j]), j++);*/

	if (threadIdx.x < 256)
		atomicAdd(TAB + threadIdx.x, tab[threadIdx.x]);
}
