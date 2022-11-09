#pragma once
#include "lab4.cuh"
#include "device_launch_parameters.h"

#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include <map>

#define block_size 32

__device__ unsigned int counter = 0;

__host__ void GPU_compress_bmp(unsigned int* pixel_map, char* gpu_compressed_img, unsigned int* gpu_palette, int img_height, int img_width, int* palette_size);

__host__ inline void RGB_to_Int(unsigned int& RGB, int red, int green, int blue);

__host__ inline void Int_to_RGB(int RGB, char& red, char& green, char& blue);

__host__ bool compare_palettes(unsigned int* cpu_palette, unsigned int* gpu_palette, int gpu_palette_size, int cpu_palette_size);

__host__ bool compare_compressed_images(unsigned int* cpu_palette, char* cpu_compressed_img, unsigned int* gpu_palette, char* gpu_compressed_img, int img_height, int img_width);

__global__ void find_unique_colors(unsigned int* pixel_map, int* colors_checker, int img_height, int img_width);

__global__ void create_palette(int* colors_checker, unsigned int* gpu_palette, int* palette_size);

__global__ void apply_palette(unsigned int* pixel_map, char* gpu_compressed_img, unsigned int* gpu_palette, int* palette_size, int img_height, int img_width);

//__device__ void sync_device();

__host__ void lab4()
{
	std::string filename = "parrotsBIG.bmp";
	std::ifstream bmp_file(filename, std::ifstream::binary);

	BITMAPFILEHEADER bmp_file_header;
	BITMAPINFOHEADER bmp_info_header;

	bmp_file.read(reinterpret_cast<char*>(&bmp_file_header), sizeof(bmp_file_header));
	bmp_file.read(reinterpret_cast<char*>(&bmp_info_header), sizeof(bmp_info_header));

	int padding = (4 - (bmp_info_header.biWidth * (bmp_info_header.biBitCount / 8)) % 4) & 3;

	int img_height = bmp_info_header.biHeight;
	int img_width = bmp_info_header.biWidth;

	int mask = (1 << bmp_info_header.biBitCount) - 1;

	unsigned int* pixel_map = new unsigned int[img_height * img_width]{ 0 };
	RGBTRIPLE RGB;
	int unique_colors_counter = 0;
	std::map<unsigned int, int> palette;
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			bmp_file.read(reinterpret_cast<char*>(&RGB), sizeof(RGB));
			RGB_to_Int(pixel_map[i * img_width + j], RGB.rgbtRed, RGB.rgbtGreen, RGB.rgbtBlue);
		}

		bmp_file.seekg(padding, std::ios_base::cur);
	}

	auto cpu_beg = std::chrono::steady_clock().now();
	char* cpu_compressed_img = new char [img_height * img_width];
	unsigned int* cpu_palette = nullptr;
////#pragma omp parallel
//	{
////#pragma omp for
//		for (int i = 0; i < img_height; i++)
//		{
//			for (int j = 0; j < img_width; j++)
//			{
//				if (palette.find(pixel_map[i * img_width + j]) == palette.end())
//				{
//					palette[pixel_map[i * img_width + j]] = unique_colors_counter;
//					unique_colors_counter++;
//				}
//			}
//		}
//
////#pragma omp for
//		for (int i = 0; i < img_height; i++)
//		{
//			for (int j = 0; j < img_width; j++)
//			{
//				cpu_compressed_img[i * img_width + j] = palette[pixel_map[i * img_width + j]];
//			}
//		}
//
//		cpu_palette = new unsigned int[unique_colors_counter];
//		for (auto c : palette)
//		{
//			cpu_palette[c.second] = c.first;
//		}
//	}
//	auto cpu_end = std::chrono::steady_clock().now();
//	std::cout << "cpu time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_beg).count() << std::endl;

	char* gpu_compressed_img = new char[img_height * img_width]{ 0 };

	unsigned int* gpu_palette = new unsigned int[256]{ 0 };
	int gpu_palette_size = 0;
	GPU_compress_bmp(pixel_map, gpu_compressed_img, gpu_palette, img_height, img_width, &gpu_palette_size);

	//std::cout << "cpu palette size: " << palette.size() << std::endl;
	std::cout << "gpu palette size: " << gpu_palette_size << std::endl;

	//if (compare_palettes(cpu_palette, gpu_palette, gpu_palette_size, unique_colors_counter))
	//{
	//	std::cout << "palettes are same\n";
	//	if (compare_compressed_images(cpu_palette, cpu_compressed_img, gpu_palette, gpu_compressed_img, img_height, img_width))
	//		std::cout << "compressed images are same\n";
	//}



	////////////////////////////////////////////////////////////////////////////////

	/*std::ofstream file("palette_img.bmp", std::ofstream::binary);

	bmp_info_header.biClrUsed = gpu_palette_size;
	bmp_info_header.biBitCount = 8;
	bmp_info_header.biSizeImage = img_height * img_width * sizeof(char);
	bmp_file_header.bfSize = sizeof(BITMAPFILEHEADER) + bmp_info_header.biSize + gpu_palette_size * sizeof(RGBQUAD) + bmp_info_header.biSizeImage;
	bmp_file_header.bfOffBits = sizeof(BITMAPFILEHEADER) + bmp_info_header.biSize + bmp_info_header.biClrUsed * sizeof(RGBQUAD);
	unsigned char reserved = 0;
	const int paddding = (4 - ((4 - (bmp_info_header.biWidth * 3) % 4) % 4)) & 3;
	file.write(reinterpret_cast<char*>(&bmp_file_header), sizeof(BITMAPFILEHEADER));
	file.write(reinterpret_cast<char*>(&bmp_info_header), sizeof(BITMAPINFOHEADER));
	for (int i = 0; i < gpu_palette_size; i++)
	{
		char red, green, blue;
		Int_to_RGB(gpu_palette[i],red,green,blue);
		file.write(reinterpret_cast<char*>(&blue), sizeof(char));
		file.write(reinterpret_cast<char*>(&green), sizeof(char));
		file.write(reinterpret_cast<char*>(&red), sizeof(char));
		file.write(reinterpret_cast<char*>(&reserved), sizeof(char));
	}
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			file.write(reinterpret_cast<char*>(&gpu_compressed_img[i * img_width + j]), sizeof(char));
		}
		for (int k = 0; k < paddding; k++)
			file.write(reinterpret_cast<char*>(&reserved), sizeof(char));
	}

	file.close();*/





	delete[] cpu_palette;
	delete[] gpu_palette;
	delete[] gpu_compressed_img;
	delete[] cpu_compressed_img;
	delete[] pixel_map;
}

__host__ inline void RGB_to_Int(unsigned int& RGB, int red, int green, int blue)
{
	RGB = red | green << 8 | blue << 16;
}

__host__ inline void Int_to_RGB(int RGB, char& red, char& green, char& blue)
{
	int mask = (1 << 8) - 1;
	red = RGB & mask;
	green = (RGB & mask << 8) >> 8;
	blue = (RGB & mask << 16) >> 16;
}

__host__ bool compare_palettes(unsigned int* cpu_palette,unsigned int* gpu_palette, int gpu_palette_size, int cpu_palette_size)
{
	if (cpu_palette_size != gpu_palette_size)
		return false;
	for (int i = 0; i < cpu_palette_size; i++)
	{
		bool color_found = false;
		for (int j = 0; j < gpu_palette_size; j++)
			if (cpu_palette[i] == gpu_palette[j])
				color_found = true;
		if (!color_found)
			return false;
	}
	return true;
}

__host__ bool compare_compressed_images(unsigned int* cpu_palette, char* cpu_compressed_img, unsigned int* gpu_palette, char* gpu_compressed_img, int img_height, int img_width)
{
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			if (cpu_palette[cpu_compressed_img[i * img_width + j]] != gpu_palette[gpu_compressed_img[i * img_width + j]])
			{
				return false;
			}
		}
	}
	return true;
}

__host__ void GPU_compress_bmp(unsigned int* pixel_map, char* gpu_compressed_img, unsigned int* gpu_palette, int img_height, int img_width, int* palette_size)
{
	cudaEvent_t start, stop;
	float elapsed_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int* d_colors_checker = nullptr;
	unsigned int* d_pixel_map = nullptr;
	char* d_gpu_compressed_img = nullptr;
	unsigned int* d_gpu_palette = nullptr;
	int* d_palette_size = nullptr;

	cudaMalloc((void**)&d_colors_checker, 10000 * sizeof(int));
	//cudaMalloc((void**)&d_colors_checker, 256 * 256 * 256 * sizeof(bool));
	cudaMalloc((void**)&d_pixel_map, img_height * img_width * sizeof(unsigned int));
	cudaMalloc((void**)&d_gpu_compressed_img, img_height * img_width * sizeof(char));
	cudaMalloc((void**)&d_gpu_palette, 256 * sizeof(unsigned int));
	cudaMalloc((void**)&d_palette_size, sizeof(int));

	cudaMemcpy(d_pixel_map, pixel_map, img_height * img_width * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_palette_size, palette_size, sizeof(int), cudaMemcpyHostToDevice);
	
	//int grid_size = img_height;

	int grid_size = std::ceil((float)img_height * (float)img_width/block_size);

	find_unique_colors << < grid_size, block_size >> > (d_pixel_map, d_colors_checker, img_height, img_width);
	cudaDeviceSynchronize();
	//create_palette << < 32, 32 >> > (d_colors_checker, d_gpu_palette, d_palette_size);
	create_palette << < std::ceil(10000.0/32.0), 32 >> > (d_colors_checker, d_gpu_palette, d_palette_size);
	cudaDeviceSynchronize();
	apply_palette << < grid_size, block_size >> > (d_pixel_map, d_gpu_compressed_img, d_gpu_palette, d_palette_size, img_height, img_width);
	cudaDeviceSynchronize();

	cudaMemcpy(gpu_compressed_img, d_gpu_compressed_img, img_height * img_width * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_palette, d_gpu_palette, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(palette_size, d_palette_size, sizeof(int), cudaMemcpyDeviceToHost);

	(*palette_size)++;

	cudaFree(d_colors_checker);
	cudaFree(d_pixel_map);
	cudaFree(d_gpu_compressed_img);
	cudaFree(d_gpu_palette);
	cudaFree(d_palette_size);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "gpu_time: " << elapsed_time << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//system("pause");
}

__global__ void find_unique_colors(unsigned int* pixel_map, int* colors_checker, int img_height, int img_width)
{

	/*int row_idx = img_width * blockIdx.x;
	for (int i = threadIdx.x; i < img_width; i += blockDim.x)
	{
		unsigned int pixel = pixel_map[row_idx + i];
		int index = (pixel * 0xDEADBEEF) >> 19;
		while (true)
		{
			int prev = atomicCAS(&colors_checker[index], 0, pixel);
			if (prev == 0 || prev == pixel)
			{
				colors_checker[index] = pixel;
				break;
			}
			index = (index + 1) & 9999;
		}
	}*/

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if (ix < img_height * img_width)
	{
		unsigned int pixel = pixel_map[ix];
		int index = (pixel * 0xDEADBEEF) >> 19;
		while (true)
		{
			int prev = atomicCAS(&colors_checker[index], 0, pixel);
			if (prev == 0 || prev == pixel)
			{
				colors_checker[index] = pixel;
				break;
			}
			index = (index + 1) & 9999;
		}
	}

	/*int row_idx = img_width * blockIdx.x;

	for (int i = threadIdx.x; i < img_width; i += blockDim.x)
		colors_checker[pixel_map[row_idx + i]] = true;*/
}

__global__ void create_palette(int* colors_checker, unsigned int* gpu_palette, int* palette_size)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if (ix < 10000)
	{
		if (colors_checker[ix] != 0)
		{
			gpu_palette[atomicAdd(palette_size, 1)] = colors_checker[ix];
		}
	}

	/*int max_size = 256 * 256 * 256;
	int elements_per_block = max_size / gridDim.x;
	int limit = (blockIdx.x + 1) * elements_per_block;
	if (limit > max_size)
		limit = max_size;
	for (int i = elements_per_block * blockIdx.x + threadIdx.x; i < limit; i += blockDim.x)
	{
		if (colors_checker[i] != 0)
		{
			int pos = atomicAdd(palette_size, 1);
			gpu_palette[pos] = i;
		}
	}*/
}

__global__ void apply_palette(unsigned int* pixel_map, char* gpu_compressed_img, unsigned int* gpu_palette, int* palette_size, int img_height, int img_width)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if (ix < img_width * img_height)
		for (int i = 0; i < 256; i++)
			if (pixel_map[ix] == gpu_palette[i])
			{
				gpu_compressed_img[ix] = i;
				break;
			}

	/*__shared__ int palette[256];

	for (int i = threadIdx.x; i < *palette_size; i += blockDim.x)
		palette[i] = gpu_palette[i];

	__syncthreads();

	int row_idx = img_width * blockIdx.x;

	for (int i = threadIdx.x; i < img_width; i += blockDim.x)
	{
		int pixel = pixel_map[row_idx + i];
		for (int j = 0; j < *palette_size; j++)
			if (palette[j] == pixel)
			{
				gpu_compressed_img[row_idx + i] = j;
				break;
			}
	}*/
}

//__device__ void sync_device()
//{
//	unsigned int old_counter;
//	__threadfence();
//
//	if (threadIdx.x == 0)
//	{
//		old_counter = atomicInc(&counter, gridDim.x - 1);
//		__threadfence();
//
//		if (old_counter != (gridDim.x - 1))
//			while (counter != 0);
//	}
//
//	__syncthreads();
//}