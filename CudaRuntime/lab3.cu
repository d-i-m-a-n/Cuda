#pragma once
#include "lab3.cuh"
#include "device_launch_parameters.h"

#include <omp.h>
#include <chrono>
#include <iostream>

#define block_size 16

__host__ void CPU_matrixMult(int* A, int* B, int*& C, int A_rows, int A_columns, int B_rows, int B_columns);

__host__ bool compare_matrixes(int* matr1, int* matr2, int rows, int columns);

__global__ void matrixMult(int* A, int* B, int* C, int A_columns, int B_columns);

__host__ void lab3()
{
	int A_rows;
	int A_columns;
	int B_rows;
	int B_columns;

	do
	{
		std::cout << "Matrix A rows count: ";
		std::cin >> A_rows;
		std::cout << "Matrix A columns count: ";
		std::cin >> A_columns;
		std::cout << "Matrix B rows count: ";
		std::cin >> B_rows;
		std::cout << "Matrix B columns count: ";
		std::cin >> B_columns;

		if (A_columns != B_rows)
			std::cout << "Wrong matrixes dimention\n";

	} while (A_columns != B_rows);

	if (A_rows % block_size)
		A_rows += block_size - A_rows % block_size;
	if (A_columns % block_size)
		A_columns += block_size - A_columns % block_size;
	if (B_rows % block_size)
		B_rows += block_size - B_rows % block_size;
	if (B_columns % block_size)
		B_columns += block_size - B_columns % block_size;

	int* h_A = new int[A_rows * A_columns];
	int* h_B = new int[B_rows * B_columns];
	int* h_C = new int[A_rows * B_columns];

	for (int i = 0; i < A_rows * A_columns; h_A[i] = rand(), i++);

	for (int i = 0; i < B_rows * B_columns; h_B[i] = rand(), i++);

	int* CPU_mult_res = nullptr;
	auto cpu_beg = std::chrono::steady_clock().now();
	CPU_matrixMult(h_A, h_B, CPU_mult_res, A_rows, A_columns, B_rows, B_columns);
	auto cpu_end = std::chrono::steady_clock().now();
	std::cout << "cpu time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_beg).count() << std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	size_t Asize = A_rows * A_columns * sizeof(int);
	size_t Bsize = B_rows * B_columns * sizeof(int);
	size_t Csize = A_rows * B_columns * sizeof(int);

	int* d_A = NULL;
	cudaMalloc((void**)&d_A, Asize);

	int* d_B = NULL;
	cudaMalloc((void**)&d_B, Bsize);

	int* d_C = NULL;
	cudaMalloc((void**)&d_C, Csize);
	cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock = dim3(block_size, block_size);
	dim3 blocksPerGrid = dim3(B_columns / block_size, A_rows / block_size);

	matrixMult << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, A_columns, B_columns);

	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);

	std::cout << "GPU time: " << elapsed_time << std::endl;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();

	if (compare_matrixes(h_C, CPU_mult_res, A_rows, B_columns))
	{
		std::cout << "Matrixes are equal\n";
	}

	delete[] h_A;
	delete[] h_C;
	delete[] h_B;
	delete[] CPU_mult_res;
}

__global__ void matrixMult(int* A, int* B, int* C, int A_columns, int B_columns)
{

	int aBegin = A_columns * blockDim.y * blockIdx.y;
	int aEnd = aBegin + A_columns - 1;
	int aStep = blockDim.x;
	int bBegin = blockDim.x * blockIdx.x;
	int bStep = blockDim.y * B_columns;

	__shared__ int as[block_size][block_size];
	__shared__ int bs[block_size][block_size];

	int sum = 0.0;
	for (int ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep)
	{
		as[threadIdx.y][threadIdx.x] = A[ia + A_columns * threadIdx.y + threadIdx.x];
		bs[threadIdx.y][threadIdx.x] = B[ib + B_columns * threadIdx.y + threadIdx.x];

		__syncthreads();

		for (int k = 0; k < blockDim.x; k++)
			sum += as[threadIdx.y][k] * bs[k][threadIdx.x];

		__syncthreads();
	}

	int ind = B_columns * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

	C[ind] = sum;
}

__host__ void CPU_matrixMult(int* A, int* B, int*& C, int A_rows, int A_columns, int B_rows, int B_columns)
{
	C = new int[A_rows * B_columns]{ 0 };

#pragma omp parallel for
	for (int i = 0; i < A_rows; i++)
	{
		for (int j = 0; j < B_columns; j++)
		{
			for (int k = 0; k < A_columns; k++)
				C[i * A_columns + j] += A[i * A_columns + k] * B[k * B_columns + j];
		}
	}
}

__host__ bool compare_matrixes(int* matr1, int* matr2, int rows, int columns)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			if (matr1[i * rows + j] != matr2[i * rows + j])
				return false;
		}
	}
	return true;
}