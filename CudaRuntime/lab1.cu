#pragma once
#include "lab1.cuh"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

#define block_size 32

__host__ void gen_vector(int*& vec, int size);

__host__ void GPU_subtraction(int* c, const int* a, const int* b, int size);

__host__ bool compare_vectors(int* vec1, int* vec2, int size);

__host__ void CPU_subtraction(int* c, const int* a, const int* b, int size);

__global__ void subtraction_kernel(int* c, const int* a, const int* b);

__host__ void lab1()
{
    int vector_size;
    std::cout << "Vectors size: ";
    std::cin >> vector_size;
    
    if (vector_size % block_size)
        vector_size += block_size - vector_size % block_size;

    int* vec1 = nullptr;
    int* vec2 = nullptr;
    int* res_vec1 = new int[vector_size];
    int* res_vec2 = new int[vector_size];

    gen_vector(vec1, vector_size);
    gen_vector(vec2, vector_size);

    auto cpu_beg = std::chrono::steady_clock().now();
    CPU_subtraction(res_vec1, vec1, vec2, vector_size);
    auto cpu_end = std::chrono::steady_clock().now();
    std::cout << "cpu time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_beg).count() << std::endl;

    GPU_subtraction(res_vec2, vec1, vec2, vector_size);

    if(compare_vectors(res_vec1, res_vec2, vector_size))
    {
        std::cout << "vectors are equal\n";
    }

    cudaDeviceReset();
    
    delete[] vec1;
    delete[] vec2;
    delete[] res_vec1;
    delete[] res_vec2;
}

__host__ void gen_vector(int*& vec, int size)
{
    vec = new int[size];
    for (int i = 0; i < size; vec[i] = rand(), i++);
}

__host__ bool compare_vectors(int* vec1, int* vec2, int size)
{
    for (int i = 0; i < size; i++)
        if (vec1[i] != vec2[i])
            return false;
    return true;
}

__host__ void CPU_subtraction(int* c, const int* a, const int* b, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] - b[i];
    }
}

__host__ void GPU_subtraction(int* c, const int* a, const int* b, int size)
{
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    cudaEvent_t start, stop;
    float elapsed_time;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    
    int grid_size = ceil((double)size / (double)block_size);

    subtraction_kernel << <grid_size, block_size >> > (dev_c, dev_a, dev_b);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "gpu_time: " << elapsed_time << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void subtraction_kernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] - b[i];
}
