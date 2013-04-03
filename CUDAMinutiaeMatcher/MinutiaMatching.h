#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "DirectionalFiltering.h"

// GPU FUNCITONS

const int DistanceToleranceBox = 9;
const int MatchingToleranceBox = 36;
const int AngleToleranceBox = CUDART_PI_F/8;

__device__ int DetermineLengthSquare(int dx, int dy)
{
	return dx*dx+dy*dy;
}

__global__ void MatchMinutiae(CUDAArray<float> result, CUDAArray<int> X1, CUDAArray<int> X2, CUDAArray<int> Y1, CUDAArray<int> Y2)
{
	__shared__ int x1[32][32];
	__shared__ int y1[32][32];
	__shared__ int x2[32][32];
	__shared__ int y2[32][32];

	//each shared row corresponds to the fprint centered at its index's minutia
	int dx = X1.At(0,threadIdx.x);
	x1[threadIdx.x][threadIdx.y] = X1.At(0,threadIdx.y)-dx;
	dx = Y1.At(0,threadIdx.x);
	y1[threadIdx.x][threadIdx.y] = Y1.At(0,threadIdx.y)-dx;
	dx = X2.At(0,threadIdx.x);
	x2[threadIdx.x][threadIdx.y] = X2.At(0,threadIdx.y)-dx;
	dx = Y2.At(0,threadIdx.x);
	y2[threadIdx.x][threadIdx.y] = Y2.At(0,threadIdx.y)-dx;

	__syncthreads();

	// now threadidx.x is the row for the 1st, threadidx.y - for second
	int max=0;
	for(int i=0;i<32;i++)
	{
		if(i==threadIdx.x)continue;

		int length = DetermineLengthSquare(x1[threadIdx.x][i],y1[threadIdx.x][i]);
		float angle = atan2((float)-y1[threadIdx.x][i],(float)x1[threadIdx.x][i]);
		for(int j=0;j<32;j++)
		{
			if(j==threadIdx.y)continue;

			int length2 = DetermineLengthSquare(x2[threadIdx.y][j],y2[threadIdx.y][j]);

			if (abs(length - length2) > DistanceToleranceBox) continue;

			float angle2 = atan2((float)-y2[threadIdx.x][i],(float)x2[threadIdx.y][j]);

			if (abs(angle - angle2) > AngleToleranceBox) continue;

			// do fancy stuff

			float cos = cosf(angle2 - angle);
			float sin = -sinf(angle2 - angle);
			int mask = 0;
			int count=0;
			for(int m =0; m<32;m++)
			{
				float xDash = cos * x1[threadIdx.x][m] - sin * y1[threadIdx.x][m];
                float yDash = sin * x1[threadIdx.x][m] + cos * y1[threadIdx.x][m];

				int nMax = -1;
				float dMax = 100500.0f;

				for(int n=0;n<32;n++)
				{
					float d = (xDash - x2[threadIdx.y][n]) * (xDash - x2[threadIdx.y][n]) + (yDash - y2[threadIdx.y][n]) * (y2[threadIdx.y][n]);
					if(d<MatchingToleranceBox&&d<dMax&&((mask>>n)&1==0))
					{
						dMax = d;
						nMax = n;
					}
				}

				if(nMax!=-1)
				{
					mask = mask | (1<<nMax);
					count++;
				}
			}

			if(count>max)max=count;
		}
	}

	result.SetAt(threadIdx.x,threadIdx.y,max);
}

// CPU FUNCTIONS

void MatchFingers(CUDAArray<int> x1, CUDAArray<int> y1, CUDAArray<int> x2, CUDAArray<int> y2)
{
	CUDAArray<float> result = CUDAArray<float>(32,32);

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(1,1);

	MatchMinutiae<<<gridSize, blockSize>>>(result, x1,y1,x2,y2);

	cudaError_t error;
	error = cudaDeviceSynchronize();
	result.Dispose();
}