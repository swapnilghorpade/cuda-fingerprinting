
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "math_functions.h"
#include "device_launch_parameters.h"
#include <time.h>

extern "C"
{
	__declspec(dllexport) int *BitCounter(int* arr, int x, int y);
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

__global__ void CountBitsInMatrix(int *arr, int x, int y, int pitch, int *res)
		{

			int sum = 0;
			
			const int length = pitch * y;
			
			int Id = threadIdx.x + blockIdx.x * blockDim.x;
			
			if (Id < x)
			{
				int accum = 0;
				int tmp = Id;
				while (tmp < length)
				{
					accum += CountBits(arr[tmp]);
					tmp += pitch;
				}
				res[Id] = accum;
			}
		}
	 
	int *BitCounter(int* arr, int x, int y)
		{
			int *res, *resDev, *arrDev; 
			size_t pitch;
			
			cudaError_t status;
			status = cudaMalloc( (void**)&resDev, x * sizeof(int) );
			status = cudaMallocPitch((void**)&arrDev, &pitch, x * sizeof(int), y);
			
			status = cudaMemcpy2D(arrDev, pitch, arr, x * sizeof(int), x * sizeof(int), y, cudaMemcpyHostToDevice);
			pitch /= sizeof(int);

			dim3 amountBlock = dim3((x + 1023) / 1024);
			dim3 amountTrds = dim3(1024);
			CountBitsInMatrix<<<amountBlock, amountTrds>>>(arrDev, x, y, pitch, resDev);
			status = cudaDeviceSynchronize();
			res = (int*)malloc( x * sizeof(int));
			cudaMemcpy(res, resDev, x * sizeof(int), cudaMemcpyDeviceToHost);
			
			cudaFree(resDev);
			cudaFree(arrDev);
			
			return res;
		}
	
	main()
	{
		int x = 1000000;
		int y = 48;
		int *arr;
		arr = (int *)malloc((x * y) * sizeof(int));
		int twoInPow = 0;
		for (int i = 0; i < 12; i++)
		{
			twoInPow = twoInPow * 2 + 1;
			for (int j = 0; j < 4000000; j++)
				arr[i * 4000000 + j] = twoInPow;
		}
		
		//clock_t start = clock();
		
		int *arrOut = BitCounter(arr, x, y);
		
		//clock_t end = clock();
		//printf("%d\n", end - start);
		//printf("%d\n", CLOCKS_PER_SEC);
		//printf("%.3f\n", ((float)end - start) / CLOCKS_PER_SEC);

		printf("%d\n",arrOut[10000]);
		
		
		system("pause");
	}