
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include "math_functions.h"

#define ceilMod(x, y) (x+y-1)/y
extern "C" {
__declspec(dllexport) void Comparing(int *data, int dataCount, int *sample, int sampleCount, int *maskData, int *maskSample, int *offset, int offsetCount, float *result, int numberOfBlocks);
}


__device__ int CountBits(int x)
{
	x -= (x >> 1) & (0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
	x = (x & 0x0000FFFF) + (x >> 16);
	return x;
}


__global__ void GetScore(int *dev_data, int dev_dataCount, int *dev_sample, int dev_sampleCount, int *dev_maskData, int *dev_maskSample, float *dev_result, size_t pitch, int numberOfBlocks) {
	int x = blockIdx.y;
	int y = blockIdx.x* blockDim.x + threadIdx.x;
	int Width = pitch / sizeof(float);	
	if ((x < dev_sampleCount/numberOfBlocks) && (y < dev_dataCount/numberOfBlocks)) {
		int unityCountOfXOR = 0;
		int unityCountOfData = 0;
		int unityCountOfSample = 0;
		for (int i = 0 ; i < numberOfBlocks ; i ++) {
			int mask = dev_maskData[numberOfBlocks*y+i] & dev_maskSample[numberOfBlocks*x+i];
			int temp1 = dev_data[numberOfBlocks*y+i] & mask;
			int temp2 = dev_sample[numberOfBlocks*x+i] & mask;
			unityCountOfXOR += CountBits(temp1^temp2);
			unityCountOfData += CountBits(temp1);
			unityCountOfSample += CountBits(temp2);
		}
		if ((unityCountOfData == 0) && (unityCountOfSample == 0))
			dev_result[x * Width + y] = 0;
		else
			dev_result[x * Width + y] = 1 - ((float) unityCountOfXOR) / (float)(unityCountOfData + unityCountOfSample);
	}
}
		


void Comparing(int *data, int dataCount, int *sample, int sampleCount, int *maskData, int *maskSample, int *offset, int offsetCount, float *result, int numberOfBlocks)
{
	int *dev_data, *dev_sample, *dev_maskData, *dev_maskSample, *dev_offset;
	float *dev_result;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
		
	size_t pitch;
	cudaStatus = cudaMallocPitch((void**)&dev_result, &pitch, (dataCount/numberOfBlocks)*sizeof(float), sampleCount/numberOfBlocks);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPitch!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_data, sizeof(int)*dataCount);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }
	
	cudaStatus = cudaMalloc((void**)&dev_sample, sizeof(int)*sampleCount);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_maskData, sizeof(int)*dataCount);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_maskSample, sizeof(int)*sampleCount);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_offset, sizeof(int)*offsetCount);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_data, data, dataCount*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_maskData, maskData, dataCount*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}

	cudaStatus = cudaMemcpy(dev_sample, sample, sampleCount*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}

	cudaStatus = cudaMemcpy(dev_maskSample, maskSample, sampleCount*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}

	cudaStatus = cudaMemcpy(dev_offset, offset , offsetCount*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}

	GetScore<<< dim3(ceilMod(dataCount/numberOfBlocks, 1024), sampleCount/numberOfBlocks), 1024 >>>(dev_data, dataCount, dev_sample, sampleCount, dev_maskData, dev_maskSample, dev_result, pitch,numberOfBlocks);
	cudaStatus = cudaMemcpy2D(result, (dataCount/numberOfBlocks)*sizeof(float), dev_result, pitch, (dataCount/numberOfBlocks)*sizeof(float), sampleCount/numberOfBlocks, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	Error:
		cudaFree(dev_data);
		cudaFree(dev_sample);
		cudaFree(dev_maskData);
		cudaFree(dev_maskSample);
		cudaFree(dev_offset);
		cudaFree(dev_result);
		cudaDeviceReset();
}
