#include "Point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma once
__global__ void Fill(bool *dev_field,Point *dev_Hull,int NHull) {
	dev_field[blockIdx.x * blockDim.x + threadIdx.x] = true;
	for (int i = NHull-1; (i>0) && (dev_field[blockIdx.x * blockDim.x + threadIdx.x]); i--)
		if ((dev_Hull[i-1].X-dev_Hull[i].X)*((int)(threadIdx.x) - dev_Hull[i].Y) - (dev_Hull[i-1].Y-dev_Hull[i].Y)*((int)(blockIdx.x) - dev_Hull[i].X) < 0)
			dev_field[blockIdx.x * blockDim.x + threadIdx.x] = false;
	if ((dev_Hull[NHull-1].X-dev_Hull[0].X)*((int)(threadIdx.x) - dev_Hull[0].Y) - (dev_Hull[NHull-1].Y-dev_Hull[0].Y)*((int)(blockIdx.x) - dev_Hull[0].X) < 0)
			dev_field[blockIdx.x * blockDim.x + threadIdx.x] = false;
}

void FieldFilling(bool *field,int rows, int columns,Point *Hull, int NHull) {
	bool *dev_field;
	cudaMalloc(&dev_field,(rows*columns)*sizeof(bool));
	Point *dev_Hull;
	cudaMalloc(&dev_Hull,NHull*sizeof(Point));
	cudaMemcpy(dev_Hull,Hull,(size_t)(NHull * sizeof(Point)), cudaMemcpyHostToDevice);
    Fill<<<rows,columns>>>(dev_field,dev_Hull,NHull);
	cudaMemcpy(field,dev_field,(size_t)((rows*columns) * sizeof(bool)), cudaMemcpyDeviceToHost);
	cudaFree(dev_field);
	cudaFree(dev_Hull);
}
