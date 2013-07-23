
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include "math_functions.h"

__device__ int CountBits(int x)
{
	x -= (x >> 1) & (0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
	x = (x & 0x0000FFFF) + (x >> 16);
	return x;
}
//__device__ int *logicFunc(int *data1, int *data2, int *mask1, int *mask2, int *output, int pitch)
//{
//	const int length = 48;
//	for (int i = 0; i < length; i++)
//	{
//		int intMask = mask1[i] & mask2[i];
//		int tmp1 = data1[i] ^ intMask;
//		int tmp2 = data2[i] ^ intMask;
//		output[i] = CountBits(tmp1 ^ tmp2) / (CountBits(tmp1) + CountBits(tmp2));
//	}
//	return output;
//}
//__global__ void ComparingCuda(int *data, int *mask1, int *dataToComp, int *mask2, int length, int pitch)
//{
//			
//	int Id = threadIdx.x + blockIdx.x * blockDim.x;
//	pitch /= sizeof(int);
//
//	if (Id < length)
//	{
//		int tmp = Id;
//		for (int i = 0; i < 48; i++)
//		{
//			int intMask = mask1[tmp] & mask2[i];
//			int tmp1 = data[tmp] ^ intMask;
//			int tmp2 = dataToComp[i] ^ intMask;
//			data[tmp] = CountBits(tmp1 ^ tmp2) / (CountBits(tmp1) + CountBits(tmp2));
//			tmp += pitch;
//		}
//	}
//}

__global__ void GetScore(int *dev_data, int dev_dataCount, int *dev_sample, int dev_sampleCount, int *dev_maskData, int *dev_maskSample, float *dev_result, size_t pitch) {
	int x = blockIdx.y;
	int y = blockIdx.x* blockDim.x + threadIdx.x;
	if ((x < dev_sampleCount/48) && (y < dev_dataCount/48)) {
		int unityCountOfXOR = 0;
		int unityCountOfData = 0;
		int unityCountOfSample = 0;
		int Width = pitch / sizeof(float);
		for (int i = 0 ; i <48 ; i ++) {
			int mask = dev_maskData[48*y+i] & dev_maskSample[48*x+i];
			int temp1 = dev_data[48*y+i] & mask;
			int temp2 = dev_sample[48*x+i] & mask;
			unityCountOfXOR += CountBits(temp1^temp2);
			unityCountOfData += CountBits(temp1);
			unityCountOfSample += CountBits(temp2);
		}
		dev_result[x * Width + y] = 1 - ((float) unityCountOfXOR) / (float)(unityCountOfData + unityCountOfSample);
	}
}
		
void Comparing(int *data, int dataCount, int *sample, int sampleCount, int *maskData, int *maskSample, int *offset, int offsetCount, float *result)
{
	int *dev_data, *dev_sample, *dev_maskData, *dev_maskSample, *dev_offset;
	float *dev_result;
	int dev_dataCount, dev_sampleCount, dev_offsetCount;
	
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
		
	size_t pitch;
	cudaStatus = cudaMallocPitch((void**)&dev_result, &pitch, (dataCount/48)*sizeof(float), sampleCount/48);
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

	GetScore<<< dim3(dataCount/(48*1024), sampleCount/48), 1024 >>>(dev_data, dev_dataCount, dev_sample, dev_sampleCount, dev_maskData, dev_maskSample, dev_result, pitch);
	
	cudaStatus = cudaMemcpy2D(result, (dataCount/48)*sizeof(float), dev_result, pitch, (dataCount/48)*sizeof(float), sampleCount/48, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	
	/*int *data1Dev, *data2Dev, *mask1Dev, *mask2Dev;
	size_t pitch;
	cudaMallocPitch((void**)*data1Dev, &pitch, sizeof(int) * length, 48);
	cudaMallocPitch((void**)*mask1Dev, &pitch, sizeof(int) * length, 48);
	cudaMalloc((void**)data2Dev, sizeof(int *));
	cudaMalloc((void**)mask2Dev, sizeof(int *));
	*/	
	//cudaMemcpy2D(data1Dev, pitch, data1, length * sizeof(int), length, 48, cudaMemcpyHostToDevice);
	//cudaMemcpy2D(mask1Dev, pitch, mask1, length * sizeof(int), length, 48, cudaMemcpyHostToDevice);
	//cudaMemcpy(data2Dev, data2, 48 * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(mask2Dev, mask2, 48 * sizeof(int), cudaMemcpyHostToDevice);

	//int AmBlocks = (length + 1023) / 1024;
	//int AmThrd = 1024;
		
	/*ComparingCuda<<<AmBlocks, AmThrd>>>(data1Dev, maskвваммеппппиины1Dev, data2Dev, mask2Dev, length, pitch);
		
	cudaMemcpy2D(data1, length, data1Dev, pitch, pitch, 48, cudaMemcpyDeviceToHost);

	cudaFree(data1Dev);
	cudaFree(data2Dev);
	cudaFree(mask1Dev);
	cudaFree(mask2Dev);
	*/

//	int *dev_data, *dev_sample, *dev_maskData, *dev_maskSample, *dev_offset;
//	float *dev_result;
//	int dev_dataCount, dev_sampleCount, dev_offsetCount;

	Error:
		cudaFree(dev_data);
		cudaFree(dev_sample);
		cudaFree(dev_maskData);
		cudaFree(dev_maskSample);
		cudaFree(dev_offset);
		//return cudaStatus;
}

int main()
{
	//int *data, int dataCount, int *sample, int sampleCount, int *maskData, int *maskSample, int *offset, int offsetCount, float *result
	cudaError_t cudaStatus;
	int dataCount = 48;
	int *data = (int*)calloc(dataCount,sizeof(int));
	int *maskData = (int*)calloc(dataCount,sizeof(int));
	maskData[0] = 1;
	data[0] = 1;

	int sampleCount = 48;
	int *sample = (int*)calloc(sampleCount,sizeof(int));
	int *maskSample = (int*)calloc(sampleCount,sizeof(int));
	maskSample[0] = 1;
	sample[0] = 1;

	int offsetCount = 1;
	int *offset = (int*)calloc(offsetCount,sizeof(int));
	offset[0] = 0;

	float *result = (float*)calloc(dataCount*sampleCount/(48*48), sizeof(float));
	
	//cudaStatus = Comparing(data,dataCount,sample,sampleCount,maskData,maskSample,offset,offsetCount,result);
	Comparing(data,dataCount,sample,sampleCount,maskData,maskSample,offset,offsetCount,result);
	printf("%f\n",result[0]);
	system("PAUSE");
}

