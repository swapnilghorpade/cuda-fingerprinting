#include "Point.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Fill(bool *dev_field,Point *dev_minutiae,int NoM) {
	dev_field[blockIdx.x * blockDim.x + threadIdx.x] = true;
	for (int i = NoM-1; (i>0) && (dev_field[blockIdx.x * blockDim.x + threadIdx.x]); i--)
		if ((dev_minutiae[i-1].X - dev_minutiae[i].X) * ((int)threadIdx.x - dev_minutiae[i].Y) - (dev_minutiae[i-1].Y - dev_minutiae[i].Y) * ((int)blockIdx.x - dev_minutiae[i].X) < 0)
			dev_field[blockIdx.x * blockDim.x + threadIdx.x] = false;
	if ((dev_minutiae[NoM-1].X - dev_minutiae[0].X) * ((int)threadIdx.x - dev_minutiae[0].Y) - (dev_minutiae[NoM-1].Y - dev_minutiae[0].Y) * ((int)blockIdx.x - dev_minutiae[0].X) < 0)
			dev_field[blockIdx.x * blockDim.x + threadIdx.x] = false;
}

void FieldFilling(bool **field,int rows, int columns,Point *Minutiae, int NoM) {
	bool *dev_field;
	for (int i = 0 ; i < NoM; i ++ )
		Minutiae[i].print();
	cudaMalloc(&dev_field,rows*columns*sizeof(bool));
	Point *dev_minutiae;
	cudaMalloc(&dev_minutiae,NoM*sizeof(Point));
	cudaMemcpy((void**)&dev_minutiae,Minutiae,NoM * sizeof(Point), cudaMemcpyHostToDevice);
    Fill<<<rows,columns>>>(dev_field,dev_minutiae,NoM);
	for (int i = 0 ; i < rows ; i++ )
		cudaMemcpy((void**) &field[i],(void**)dev_field[i*rows],columns* sizeof(bool),cudaMemcpyDeviceToHost);
}