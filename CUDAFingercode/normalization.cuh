#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "math.h"
#include "math_constants.h"
#include <stdlib.h>

__global__ void applyNormalization(
	int* image,
	float baseMean,
	float baseVariance,
	float meanIn,
	float varianceIn,
	float meanOut,
	float varianceOut,
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadiusSquared, // inner band radius
	int bandMaxRadiusSquared,  // outer band radius)
	int horizontalSize)
{
	int x = blockIdx.x*32+threadIdx.x;
	int y = blockIdx.y*32+threadIdx.y;

	float value = (float)image[x+y*horizontalSize*32];
    int result = baseMean;
	int dX = x-referenceX;
	int dY = y-referenceY;

	int distance = dX*dX+dY*dY;

	if(distance>=bandMinRadiusSquared&&distance<=bandMaxRadiusSquared)
	{
		if (varianceIn > 0)
        {
			result +=
				((value > meanIn) ? (1) : (-1))*
				(int) sqrtf(baseVariance/varianceIn*(value - meanIn)*(value - meanIn));
            if (result < 0) result = 0;
			if (result > 255.0f) result = 255;
        }
	}
	else
	{
		if (varianceOut > 0)
        {
			result +=
				((value > meanOut) ? (1) : (-1))*
				(int) sqrtf(baseVariance/varianceOut*(value - meanOut)*(value - meanOut));
            if (result < 0) result = 0;
			if (result > 255.0f) result = 255;
        }
	}

	image[x+y*horizontalSize*32] = (int)result;
}

__global__ void countVariances(
	int* image,
	float* variances,
	float meanIn,
	float meanOut,
	int squareSize,
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadiusSquared, // inner band radius
	int bandMaxRadiusSquared,  // outer band radius
	int horizontalSize,
	int verticalSize)
{
	int index = threadIdx.x+threadIdx.y*horizontalSize;
	
	int baseX = threadIdx.x*32;
	int baseY = threadIdx.y*32;

	float varianceIn =0;
	float varianceOut =0;

	for(int x=baseX;x<baseX+32;x++)
	{
		for(int y=baseY;y<baseY+32;y++)
		{
			int dX = x-referenceX;
			int dY = y-referenceY;

			int distance = dX*dX+dY*dY;
			float color = (float)image[x+y*horizontalSize*32];

			if(distance>=bandMinRadiusSquared&&distance<=bandMaxRadiusSquared)
			{
				varianceIn+= (color-meanIn)*(color-meanIn);
			}
			else
			{

				varianceOut+=(color-meanOut)*(color-meanOut);
			}
		}
	}

	int outOffset = horizontalSize*verticalSize;
	variances[index] = varianceIn;
	variances[index+outOffset]=varianceOut;
}

__global__ void countMeans(
	int* image,
	float* means,
	int* meanCounts,
	int squareSize,
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadiusSquared, // inner band radius
	int bandMaxRadiusSquared,  // outer band radius
	int horizontalSize,
	int verticalSize)
{
	int index = threadIdx.x+threadIdx.y*horizontalSize;
	
	int baseX = threadIdx.x*32;
	int baseY = threadIdx.y*32;

	float meanIn =0;
	float meanOut =0;
	float meanCountIn =0;
	float meanCountOut =0;

	for(int x=baseX;x<baseX+32;x++)
	{
		for(int y=baseY;y<baseY+32;y++)
		{
			int dX = x-referenceX;
			int dY = y-referenceY;

			int distance = dX*dX+dY*dY;

			if(distance>=bandMinRadiusSquared&&distance<=bandMaxRadiusSquared)
			{
				meanIn+=image[x+y*horizontalSize*32];
				meanCountIn++;
			}
			else
			{
				meanOut+=image[x+y*horizontalSize*32];
				meanCountOut++;
			}
		}
	}

	int outOffset = horizontalSize*verticalSize;
	means[index] = meanIn;
	means[index+outOffset]=meanOut;
	meanCounts[index] = meanCountIn;
	meanCounts[index+outOffset]=meanCountOut;
}

void normalizeImage(
	int* image, // the image itself - already stored in cuda
	int width, // its width
	int height, // height
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadius, // inner band radius
	int bandMaxRadius, // outer band radius
	int baseMean, // the base mean of normalized areas
	int baseVariance) //the base variance
{
	int* means;
	int* variances;
	int* meanCounts;
	int* varianceCounts;

	int verticalSize = height/32;
	int horizontalSize = width/32;

	int size = horizontalSize*verticalSize; //32*32
	
	float* variancesCpu = (float*)malloc(2*size*sizeof(float));
	float* meansCpu = (float*)malloc(2*size*sizeof(float));
	float* meanCountsCpu = (float*)malloc(2*size*sizeof(float));

	cudaError_t error = cudaMalloc(&means, 2*size*sizeof(float));
	error = cudaMalloc(&variances, 2*size*sizeof(float));
	error =cudaMalloc(&meanCounts, 2*size*sizeof(int));

	countMeans<<<1,dim3(horizontalSize,verticalSize)>>>(
		image,
		means,
		meanCounts,
		32,
		referenceX,
		referenceY,
		bandMinRadius*bandMinRadius,
		bandMaxRadius*bandMaxRadius,
		horizontalSize, 
		verticalSize);

	error = cudaGetLastError();

	float meanIn = 0;
	float meanInCount = 0;
	float meanOut = 0;
	float meanOutCount = 0;
	cudaMemcpy(meansCpu,means, 2*size*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(meanCountsCpu,meanCounts, 2*size*sizeof(float),cudaMemcpyDeviceToHost);
	int offset = verticalSize*horizontalSize;
	for(int i=0;i<offset;i++)
	{
		meanIn+=meansCpu[i];
		meanOut+=meansCpu[offset+i];
		meanInCount += meanCountsCpu[i];
		meanOutCount += meanCounts[i+offset];
	}

	countVariances<<<1,dim3(horizontalSize,verticalSize)>>>(
		image,
		variances,
		meanIn,
		meanOut,
		32,
		referenceX,
		referenceY,
		bandMinRadius*bandMinRadius,
		bandMaxRadius*bandMaxRadius,
		horizontalSize,
		verticalSize);

	error = cudaGetLastError();

	error = cudaMemcpy(variancesCpu,variances, 2*size*sizeof(float),cudaMemcpyDeviceToHost);
	float varianceIn = 0;
	float varianceOut = 0;
	for(int i=0;i<offset;i++)
	{
		varianceIn+=variancesCpu[i];
		varianceOut+=variancesCpu[offset+i];
	}

	applyNormalization<<<dim3(horizontalSize,verticalSize),dim3(32,32)>>>(
		image,
		100.0f,
		100.0f,
		meanIn,
		varianceIn,
		meanOut,
		varianceOut,
		referenceX,
		referenceY,
		bandMinRadius*bandMinRadius,
		bandMaxRadius*bandMaxRadius,
		horizontalSize
		);

	error = cudaGetLastError();
}