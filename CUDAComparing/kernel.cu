
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include "math_functions.h"
/*
__device__ int CountBits(int x)
        {
            x -= (x >> 1) & (0x55555555);
            x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
            x = (x + (x >> 4)) & 0x0F0F0F0F;
            x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
            x = (x & 0x0000FFFF) + (x >> 16);
            return x;
		}
__device__ int *logicFunc(int *data1, int *data2, int *mask1, int *mask2, int *output, int pitch)
		{
			const int length = 48;
            for (int i = 0; i < length; i++)
            {
                int intMask = mask1[i] & mask2[i];
                int tmp1 = data1[i] ^ intMask;
                int tmp2 = data2[i] ^ intMask;
                output[i] = CountBits(tmp1 ^ tmp2) / (CountBits(tmp1) + CountBits(tmp2));
            }
            return output;
		}
__global__ void ComparingCuda(int *data, int *mask1, int *dataToComp, int *mask2, int length, int pitch)
		{
			
			int Id = threadIdx.x + blockIdx.x * blockDim.x;
			pitch /= sizeof(int);

			if (Id < length)
			{
				int tmp = Id;
				for (int i = 0; i < 48; i++)
				{
					int intMask = mask1[tmp] & mask2[i];
					int tmp1 = data[tmp] ^ intMask;
					int tmp2 = dataToComp[i] ^ intMask;
					data[tmp] = CountBits(tmp1 ^ tmp2) / (CountBits(tmp1) + CountBits(tmp2));
					tmp += pitch;
				}
			}
		}

	void Comparing(int *data1, int *data2, int *mask1, int *mask2, int length)
	{
		int *data1Dev, *data2Dev, *mask1Dev, *mask2Dev;
		size_t pitch;
		cudaMallocPitch((void**)*data1Dev, &pitch, sizeof(int) * length, 48);
		cudaMallocPitch((void**)*mask1Dev, &pitch, sizeof(int) * length, 48);
		cudaMalloc((void**)data2Dev, sizeof(int *));
		cudaMalloc((void**)mask2Dev, sizeof(int *));
		
		cudaMemcpy2D(data1Dev, pitch, data1, length * sizeof(int), length, 48, cudaMemcpyHostToDevice);
		cudaMemcpy2D(mask1Dev, pitch, mask1, length * sizeof(int), length, 48, cudaMemcpyHostToDevice);
		cudaMemcpy(data2Dev, data2, 48 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(mask2Dev, mask2, 48 * sizeof(int), cudaMemcpyHostToDevice);

		int AmBlocks = (length + 1023) / 1024;
		int AmThrd = 1024;
		
		ComparingCuda<<<AmBlocks, AmThrd>>>(data1Dev, maskвваммеппппиины1Dev, data2Dev, mask2Dev, length, pitch);
		
		cudaMemcpy2D(data1, length, data1Dev, pitch, pitch, 48, cudaMemcpyDeviceToHost);

		cudaFree(data1Dev);
		cudaFree(data2Dev);
		cudaFree(mask1Dev);
		cudaFree(mask2Dev);

	}
*/
int main()
{
}

