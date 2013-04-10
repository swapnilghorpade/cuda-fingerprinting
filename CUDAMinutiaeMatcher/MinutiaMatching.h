#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "DirectionalFiltering.h"

// GPU FUNCITONS

const int DistanceToleranceBox = 3;
const int MatchingToleranceBox = 36;

__device__ float DetermineLength(int dx, int dy)
{
	return sqrt((float)(dx*dx+dy*dy));
}

__global__ void MatchMinutiae(CUDAArray<int> result, CUDAArray<int> X1, CUDAArray<int> Y1, CUDAArray<int> X2, CUDAArray<int> Y2)
{
	__shared__ int x1[32][32];
	__shared__ int y1[32][32];
	__shared__ int x2[32][32];
	__shared__ int y2[32][32];

	__shared__ float length1[32][32];
	__shared__ float length2[32][32];

	__shared__ float angle1[32][32];
	__shared__ float angle2[32][32];

	__shared__ unsigned int taskCount;
	__shared__ int tasks[4000];
	//each shared row corresponds to the fprint centered at its index's minutia
	int dx = X1.At(0,threadIdx.x);
	x1[threadIdx.x][threadIdx.y] = X1.At(0,threadIdx.y)-dx;
	dx = Y1.At(0,threadIdx.x);
	y1[threadIdx.x][threadIdx.y] = Y1.At(0,threadIdx.y)-dx;
	dx = X2.At(blockIdx.x,threadIdx.x);
	x2[threadIdx.x][threadIdx.y] = X2.At(blockIdx.x,threadIdx.y)-dx;
	dx = Y2.At(blockIdx.x,threadIdx.x);
	y2[threadIdx.x][threadIdx.y] = Y2.At(blockIdx.x,threadIdx.y)-dx;

	length1[threadIdx.x][threadIdx.y] = DetermineLength(x1[threadIdx.x][threadIdx.y], y1[threadIdx.x][threadIdx.y]);
	length2[threadIdx.x][threadIdx.y] = DetermineLength(x2[threadIdx.x][threadIdx.y], y2[threadIdx.x][threadIdx.y]);

	angle1[threadIdx.x][threadIdx.y] = atan2((float)-y1[threadIdx.x][threadIdx.y],(float)x1[threadIdx.x][threadIdx.y]);
	angle2[threadIdx.x][threadIdx.y] = atan2((float)-y2[threadIdx.x][threadIdx.y],(float)x2[threadIdx.x][threadIdx.y]);

	if(threadIdx.x==0&&threadIdx.y==0)
	{
		taskCount =0;
	}

	__syncthreads();

	// now threadidx.x is the row for the 1st, threadidx.y - for second
	for(int i=0;i<32;i++)
	{
		if(i==threadIdx.x)continue;

		for(int j=0;j<32;j++)
		{
			if(j==threadIdx.y)continue;

			if (abs(length1[threadIdx.x][i] - length2[threadIdx.y][j]) <= DistanceToleranceBox
				&&abs(angle1[threadIdx.x][i] - angle2[threadIdx.y][j]) <= CUDART_PI_F/8) 
			{
			// do fancy stuff
				unsigned int localIndex = atomicInc(&taskCount ,4000);
				tasks[localIndex-1] = (threadIdx.x<<24)|(i<<16)|(threadIdx.y<<8)|j;
			}
		}
	}

	__syncthreads();

	/*int maxCount =0;

	int limit = taskCount / 1024 + 1;

	for(int i=0; i<limit; i++)
	{
		int index = i*1024+threadIdx.x*32+threadIdx.y;
		if(index < taskCount)
		{
			int task = tasks[index];
			int m1From = task>>24;
			int m1To = (task>>16)&8;
			int m2From = (task>>8)&255;
			int m2To = task&255;

			float cosine = cos(angle2[m2From][m2To] - angle1[m1From][m1To]);
			float sine = -sin(angle2[m2From][m2To] - angle1[m1From][m1To]);
			int mask = 0;
			int count=0;
			for(int m =0; m<32;m++)
			{
				float xDash = cosine * x1[m1From][m] - sine * y1[m1From][m];
				float yDash = sine * x1[m1From][m] + cosine * y1[m1From][m];

				short nMax = -1;
				float dMax = MatchingToleranceBox;

				for(int n=0;n<32;n++)
				{
					int dX = xDash - x2[m2From][n];
					int dY = yDash - y2[m2From][n];
					int d = dX*dX+dY*dY;
					if(d<dMax&&(mask&(1<<n)==0))
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

			if(count>maxCount)
				maxCount=count;
		}
	}

	x1[threadIdx.x][threadIdx.y] = maxCount;

	__syncthreads();

	if(threadIdx.y == 0)
	{
		for(int i=1;i<32;i++)
		{
			if(x1[0][threadIdx.x]<=x1[i][threadIdx.x])
			{
				x1[0][threadIdx.x]=x1[i][threadIdx.x];
			}
		}
	}

	__syncthreads();

	if(threadIdx.x==0&&threadIdx.y==0)
	{
		int absolutemax = 0;
		for(int i=0;i<32;i++)
		{
			if(x1[0][i]>absolutemax)
			{
				absolutemax = x1[0][i];
			}
		}
			
		result.SetAt(0,blockIdx.x,absolutemax);
	}*/
}

// CPU FUNCTIONS

void MatchFingers(CUDAArray<int> x1, CUDAArray<int> y1, CUDAArray<int> x2, CUDAArray<int> y2)
{
	int n = 10;
	CUDAArray<int> result = CUDAArray<int>(n,1);

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(n,1);

	MatchMinutiae<<<gridSize, blockSize>>>(result, x1,y1,x2,y2);

	cudaError_t error = cudaDeviceSynchronize();
	cudaError_t error2 = cudaGetLastError();
	int* res = result.GetData();
	int m =0;
	for(int i=0;i<n;i++)if(res[i]>m)m=res[i];
	result.Dispose();
}