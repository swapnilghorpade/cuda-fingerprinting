#include "cuda_runtime.h"
#include <stdlib.h>
#include "CUDAArray.h"
#include "Gaussian.h"
#include "device_launch_parameters.h"

__global__ void resizeArray(CUDAArray<float> target, CUDAArray<float> source, float cellSize, float sigma)
{
	int column = defaultColumn();
	int row = defaultRow();

	if(target.Width>column&&target.Height>row)
	{
		float x = cellSize*column;
		float y = cellSize*row;

		float sum=0;
		float filterSum=0;

		for(int columnSource = (int)x-5;columnSource<=(int)x+5;columnSource++)
		{
			if(columnSource<0)continue;
			if(columnSource>=source.Width)break;

			for(int rowSource = (int)y-5;rowSource<=(int)y+5;rowSource++)
			{
				if(rowSource<0)continue;
				if(rowSource>=source.Height)break;

				float filterValue = Gasussian2D(x-columnSource, y-rowSource, sigma);
				filterSum += filterValue;
				float value=source.At(rowSource, columnSource);
				sum+= value*filterValue;
			}
		}

		target.SetAt(row, column, sum/filterSum);
	}
}

CUDAArray<float> Reduce(CUDAArray<float> source, float factor)
{
	int width = (int)((float)source.Width/factor);
	int height = (int)((float)source.Height/factor);
	CUDAArray<float> target = CUDAArray<float>(width, height);

	float sigma = factor/2.0f*0.75f;

	dim3 blocks = dim3(ceilMod(target.Width,32),ceilMod(target.Height,32));

	dim3 threads = dim3(32,32);

	resizeArray<<<blocks, threads>>>(target, source, factor, sigma);
	cudaError_t error = cudaGetLastError();
	return target;
}

CUDAArray<float> Expand(CUDAArray<float> source, float factor, int width, int height)
{
	int realWidth = width==0||height==0?(int)((float)source.Width*factor):width;
	int realHeight = width==0||height==0?(int)((float)source.Height*factor):height;
	
	CUDAArray<float> target = CUDAArray<float>(realWidth, realHeight);

	float sigma = factor/2.0f*0.75f;

	dim3 blocks = dim3(ceilMod(target.Width,32),ceilMod(target.Height,32));

	dim3 threads = dim3(32,32);

	resizeArray<<<blocks, threads>>>(target, source, 1.0f/factor, sigma);
	cudaError_t error = cudaGetLastError();
	return target;
}