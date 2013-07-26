//#include "cuda_runtime.h"
#include "math_functions.h"
#include "math_constants.h"
#include "math.h"
#include "CUDAArray.cuh"
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "ConvolutionHelper.cuh"

static int defaultThreadCount = 32;

__host__ __device__ float Gaussian2D(float x, float y, float sigma)
{
	float commonDenom = sigma*sigma*2.0f;
	float denominator = commonDenom*CUDART_PI_F;
	return expf(-(x*x+y*y)/commonDenom)/denominator;
}

__host__ __device__ float Gaussian1D(float x, float sigma)
{
	float commonDenom = sigma*sigma*2.0f;
	float denominator = sigma*sqrtf(CUDART_PI_F*2.0f);
	return expf(-(x*x)/commonDenom)/denominator;
}

__global__ void cudaConvolve(float* target, float* source, int width, int height, size_t pitch,CUDAArray<float> filter)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(width>column&&height>row)
	{
		int tX = threadIdx.x;
		int tY = threadIdx.y;
		__shared__ float filterCache[32*32];
		
		if(tX<filter.Width&&tY<filter.Height)
		{
			int indexLocal = tX+tY*filter.Width;
			filterCache[indexLocal] = filter.At(tY,tX);
		}
		__syncthreads();

		int center = filter.Width/2;

		float sum = 0.0f;

		for(int drow=-center;drow<=center;drow++)
		{
			for(int dcolumn=-center;dcolumn<=center;dcolumn++)
			{
				float filterValue1 = filterCache[filter.Width*(drow+center)+dcolumn+center];

				int valueRow = row+drow;
				if(valueRow<0)valueRow=0;
				if(valueRow>=height)valueRow = height-1;

				int valueColumn = column+dcolumn;
				if(valueColumn<0)valueColumn=0;
				if(valueColumn>=width)valueColumn = width-1;

				float value = source[valueRow*pitch+valueColumn];
				sum+=filterValue1*value;
			}
		}

		target[row*pitch + column] = sum;
	}
}


CUDAArray<float> MakeDifferentialGaussianKernel(float kx, float ky, float c, float sigma)
{
	int size = 2*(int)ceil(sigma*3.0f)+1;
	int center=size/2;
	float* kernel = (float*)malloc(sizeof(float)*size*size);
	float sum=0;
	for(int row=-center; row<=center; row++)
	{
		for(int column=-center; column<=center; column++)
		{
			sum+= kernel[column+center+(row+center)*size] = Gaussian2D(column,row,sigma)*(kx*column+ky*row+c);
		}
	}
	if (abs(sum) >0.00001f)
	for(int row=-center; row<=center; row++)
	{
		for(int column=-center; column<=center; column++)
		{
			kernel[column+center+(row+center)*size]/=sum;
		}
	}

	CUDAArray<float> cudaKernel = CUDAArray<float>(kernel,size,size);

	free(kernel);

	return cudaKernel;
}

void Convolve(float* target, float* source, int width, int height, size_t pitch, CUDAArray<float> filter)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(width,defaultThreadCount),
		ceilMod(height,defaultThreadCount));

	cudaConvolve<<<gridSize,blockSize>>>(target, source, width, height, pitch, filter);
	cudaError_t error  = cudaDeviceSynchronize();
	int i=0;
}

void smoothing(float* smoothed, float* source, int width, int height, size_t pitch)
{
	float sigma = 1.4;//factor/2.0f*0.75f;
	
	CUDAArray<float> kernel = MakeDifferentialGaussianKernel(0, 0, 1, sigma);

	Convolve(smoothed, source, width, height, pitch, kernel);

	kernel.Dispose();

//	return smoothed;
}