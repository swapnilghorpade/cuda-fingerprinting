#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FieldFilling.h"

__device__ int fmax(int a,int b) {
	int result = a;
	if (a < b)
		result = b;
	return result;
}


__device__ int fmin(int a,int b) {
	int result = a;
	if (a > b)
		result = b;
	return result;
}


__global__ void FindArea(bool *dev_field,bool *dev_NewField,int radius,int rows,int columns) {
	for (int i = fmax(blockIdx.x-radius,0); (i <= fmin(rows-1,blockIdx.x+radius)) && (!dev_NewField[blockIdx.x*blockDim.x+threadIdx.x]); i++)
		for (int j = fmax(threadIdx.x-radius,0); (j <= fmin(columns-1,threadIdx.x+radius)) && (!dev_NewField[blockIdx.x*blockDim.x+threadIdx.x]); j++) 
			if ((threadIdx.x-j)*(threadIdx.x-j) + (blockIdx.x-i)*(blockIdx.x-i) <= radius * radius)
				if (dev_field[i*blockDim.x+j])
					dev_NewField[blockIdx.x*blockDim.x+threadIdx.x] = true;
}


void BuildWorkingArea(bool *field,int rows,int columns,int radius) {
	bool *dev_field;
	bool *dev_NewField;
	cudaMalloc(&dev_field,rows*columns*sizeof(bool));
	cudaMalloc(&dev_NewField,rows*columns*sizeof(bool));
	cudaMemcpy(dev_field,field,(size_t)(rows * columns * sizeof(bool)), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_NewField,field,(size_t)(rows * columns * sizeof(bool)), cudaMemcpyHostToDevice);
	cudaSetDevice(0);
	FindArea<<<rows,columns>>>(dev_field,dev_NewField,radius,rows,columns);
	cudaDeviceReset();
	cudaMemcpy(field,dev_NewField,(size_t)(rows * columns * sizeof(bool)), cudaMemcpyDeviceToHost);
	cudaFree(dev_field);
	cudaFree(dev_NewField);
}