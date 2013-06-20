#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "MinutiaMatching.h"

__device__ const float tauPS = 0.2f;
__device__ const int NeighborhoodSize = 11;
__device__ const float tauLS = 0.8f;

// GPU FUNCTIONS

__global__ void cudaEstimateMeasure(CUDAArray<float> psiReal, CUDAArray<float> psiIm, 
	CUDAArray<float> lsM, CUDAArray<float> psReal, CUDAArray<float> psIm)
{
	int row = defaultRow();
	int column = defaultColumn();
	if(lsM.Width>column&&lsM.Height>row)
	{
		float lsValue = lsM.At(row,column);
		float psImValue = psIm.At(row,column);
		float psRealValue = psReal.At(row,column);
		
		float psMagn = psImValue*psImValue+psRealValue*psRealValue;

		float psiRealValue = psRealValue*(1.0f-lsValue);
		float psiImValue = psImValue*(1.0f-lsValue);

		psiReal.SetAt(row,column,psiRealValue);
		psiIm.SetAt(row,column,psiImValue);
	}
}

__global__ void voteForTheMax(CUDAArray<int> votesRows,CUDAArray<int> votesColumns, CUDAArray<float> psim)
{
	int row = defaultRow();
	int column = defaultColumn();
	float maxM = tauPS; // a hacky way to get only values above the treshold
	if(psim.Width>column&&psim.Height>row)
	{
		for (int dRow = -NeighborhoodSize/2; dRow <= NeighborhoodSize/2; dRow++)
		{
			int correctRow = row + dRow;
			if(correctRow > 9 && correctRow < psim.Height-10)
			{
				for (int dColumn = -NeighborhoodSize / 2; dColumn <= NeighborhoodSize / 2; dColumn++)
				{
					int correctColumn = column + dColumn;
					if(correctColumn > 9 && correctColumn < psim.Width-10)
					{
						if (psim.At(correctRow, correctColumn) > maxM)
						{
							maxM = psim.At(correctRow, correctColumn);
							votesRows.SetAt(row,column,correctRow);
							votesColumns.SetAt(row,column,correctColumn);
						}
					}
				}
			}
		}
	}
}

__global__ void calculateMinutiaMetrics(CUDAArray<int> metrics, CUDAArray<int> votes, CUDAArray<float> lsm)
{
	int row = defaultRow();
	int column = defaultColumn();

	if(lsm.Width>column&&lsm.Height>row&&votes.At(row,column) > 20)
	{
		float sum = 0;
		int count = 0;

		for (int dRow = -ringOuterRadius; dRow <= ringOuterRadius; dRow++)
		{
			for (int dColumn = -ringOuterRadius; dColumn <= ringOuterRadius; dColumn++)
			{
				if (abs(dRow) < ringInnerRadius && abs(dColumn) < ringInnerRadius) continue;
				count++;
				int correctRow = row + dRow < 0
					? 0
					: (row + dRow >= lsm.Height ? lsm.Height - 1 : row + dRow);
				int correctColumn = column + dColumn < 0
					? 0
					: (column + dColumn >= lsm.Width ? lsm.Width - 1 : column + dColumn);
				sum += lsm.At(correctRow, correctColumn);

			}
		}

		if (sum / count > tauLS)
		{
			metrics.SetAt(row,column,1);
		}
	}
}

// CPU FUNCTIONS

void EstimateMeasure(CUDAArray<float> psiReal, CUDAArray<float> psiIm, 
	CUDAArray<float> lsM, CUDAArray<float> psReal, CUDAArray<float> psIm)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(psiReal.Width, defaultThreadCount),
		ceilMod(psiReal.Height, defaultThreadCount));
	cudaEstimateMeasure<<<gridSize,blockSize>>>(psiReal, psiIm, lsM, psReal, psIm);
}

int compare( const void *arg1, const void *arg2 )
{
   return (*((float*)arg2)-*((float*)arg1))<0?-1:1;
}

void ExtractMinutiae(int** xs, int** ys, CUDAArray<float> source)
{
	SaveArray(source,"C:\\temp\\check.bin");
	clock_t clk1 = clock();
	cudaError error;
	CUDAArray<float> psReal;
	CUDAArray<float> psIm;
	CUDAArray<float> psM=CUDAArray<float>(source.Width,source.Height);
	
	CUDAArray<float> lsReal;
	CUDAArray<float> lsIm;
	CUDAArray<float> lsM=CUDAArray<float>(source.Width,source.Height);
	EstimateLS(&lsReal, &lsIm, source, 0.8f, 4.0);
	

	EstimatePS(&psReal, &psIm, source, 0.8f, 4.0f);
	GetMagnitude(lsM, lsReal, lsIm);
	GetMagnitude(psM, psReal, psIm);
	//SaveArray(psM, "C:\\temp\\check.bin");
	float* psmLocal = psM.GetData();

	CUDAArray<float> psiReal = CUDAArray<float>(lsM.Width, lsM.Height);
	CUDAArray<float> psiIm = CUDAArray<float>(lsM.Width, lsM.Height);
	CUDAArray<float> psiM = CUDAArray<float>(lsM.Width, lsM.Height);
	EstimateMeasure(psiReal, psiIm,lsM, psReal, psIm);

	GetMagnitude(psiM, psiReal, psiIm);
	
	error = cudaGetLastError();

	CUDAArray<int> votesRows = CUDAArray<int>(lsM.Width, lsM.Height);
	CUDAArray<int> votesColumns = CUDAArray<int>(lsM.Width, lsM.Height);
	CUDAArray<int> metrics = CUDAArray<int>(lsM.Width, lsM.Height);

	cudaMemset2D(votesRows.cudaPtr, votesRows.Stride, 0, votesRows.Width*sizeof(int), votesRows.Height);
	cudaMemset2D(votesColumns.cudaPtr, votesColumns.Stride, 0, votesColumns.Width*sizeof(int), votesColumns.Height);

	cudaMemset2D(metrics.cudaPtr, metrics.Stride, 0, metrics.Width*sizeof(int), metrics.Height);

	error = cudaGetLastError();

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(psiReal.Width, defaultThreadCount),
		ceilMod(psiReal.Height, defaultThreadCount));
	voteForTheMax<<<gridSize, blockSize>>>(votesRows, votesColumns, psiM);

	cudaDeviceSynchronize();
	error = cudaGetLastError();

	int* votesLocal = (int*)malloc(sizeof(int)*votesRows.Width*votesRows.Height);
	memset(votesLocal, 0, sizeof(int)*votesRows.Width*votesRows.Height);
	int* votesRowsLocal = votesRows.GetData();
	int* votesColumnsLocal = votesColumns.GetData();

	for(int i=0;i<votesRows.Width*votesRows.Height;i++)
	{
		if(votesRowsLocal[i]>0&&votesRowsLocal[i]<votesRows.Height-1&&votesColumnsLocal[i]>0&&votesColumnsLocal[i]<votesRows.Width-1)
		{
			votesLocal[votesRows.Width*votesRowsLocal[i]+votesColumnsLocal[i]]++;
		}
	}

	free(votesRowsLocal);
	free(votesColumnsLocal);
	
	CUDAArray<int> votes = CUDAArray<int>(votesLocal,votesRows.Width,votesRows.Height);

	clock_t clk2 = clock();

	calculateMinutiaMetrics<<<gridSize, blockSize>>>(metrics, votes, lsM);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	int* metricsLocal = metrics.GetData();



	lsM.Dispose();
	lsReal.Dispose();
	lsIm.Dispose();
	psReal.Dispose();
	psIm.Dispose();
	psM.Dispose();
	psiReal.Dispose();
	psiIm.Dispose();
	psiM.Dispose();
	votesRows.Dispose();
	votesColumns.Dispose();
	lsIm.Dispose();
	lsReal.Dispose();

	int count = 0;

	for(int row=1;row<metrics.Height-1;row++)
	{
		for(int column=1;column<metrics.Width-1;column++)
		{
			if(metricsLocal[row*metrics.Width+column])count++;
		}
	}

	int* minutiaeCache = (int*)malloc(count*(2*sizeof(int)+sizeof(float)));
	int cacheCount = 0;

	for(int row=1;row<metrics.Height-1;row++)
	{
		for(int column=1;column<metrics.Width-1;column++)
		{
			if(metricsLocal[row*metrics.Width+column])
			{
				bool valid = true;
				for(int i=0;i<cacheCount;i++)
				{
					int cacheColumn = minutiaeCache[3*i+1];
					int cacheRow = minutiaeCache[3*i+2];
					if((cacheRow - row)*(cacheRow - row)+(cacheColumn - column)*(cacheColumn - column) <30)
					{
						valid = false;
						break;
					}
				}

				if(valid)
				{
					minutiaeCache[3*cacheCount+1] = column;
					minutiaeCache[3*cacheCount+2] = row;
					((float*)minutiaeCache)[3*cacheCount++] = psmLocal[row*metrics.Width+column];
				}
			}
		}
	}

	qsort(minutiaeCache, cacheCount, sizeof(float)+sizeof(int)*2,compare);

	*xs = (int*)malloc(sizeof(int)*32);
	*ys = (int*)malloc(sizeof(int)*32);

	for(int i=0;i<32;i++)
	{
		(*xs)[i] = minutiaeCache[3*i+1];
		(*ys)[i] = minutiaeCache[3*i+2];
	}

	metrics.Dispose();
	votes.Dispose();
	free(metricsLocal);
	free(psmLocal);
	free(minutiaeCache);
	free(votesLocal);
}