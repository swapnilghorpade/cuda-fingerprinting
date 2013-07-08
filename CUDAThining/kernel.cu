
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

double B(double *picture, int x, int y, size_t pitch)        //Метод В(Р) возвращает количество черных пикселей в окрестности точки Р
{
	return picture[x + (y - 1)*pitch] + picture[x + 1 + (y - 1)*pitch] + picture[x + 1 + y*pitch] + picture[x + 1 + (y + 1)*pitch] +
           picture[x * (y + 1)*pitch] + picture[x - 1 + (y + 1)*pitch] + picture[x - 1 + y*pitch] + picture[x - 1 * (y - 1)*pitch];
}

double A(double *picture, int x, int y, size_t pitch)        //Метод А(Р) возвращает количество подряд идущих белых и черных пикселей вокруг точки Р (..0->1..)
{
	int counter = 0;
    if((picture[x + (y - 1)*pitch] == 0) && (picture[x + 1 + (y - 1)*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x + 1 + (y - 1)*pitch] == 0) && (picture[x + 1 + y*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x + 1 + y*pitch] == 0) && (picture[x + 1 + (y + 1)*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x + 1 + (y + 1)*pitch] == 0) && (picture[x + (y + 1)*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x + (y + 1)*pitch] == 0) && (picture[x - 1 + (y + 1)*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x - 1 + (y + 1)*pitch] == 0) && (picture[x - 1 + y*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x - 1 + y*pitch] == 0) && (picture[x - 1 + (y - 1)*pitch] == 1))
    {
        counter++;
    }
    if ((picture[x - 1 + (y - 1)*pitch] == 0) && (picture[x + (y - 1)*pitch] == 1))
    {
        counter++;
    }
    return counter;
}


__global__ double* ThiningPictureWithCUDA(double* newPicture, size_t pitch)
{
            
            double *picture = newPicture;

			int x = threadIdx.x + blockIdx.x*blockDim.x;
            int y = threadIdx.y + blockIdx.y*blockDim.y;
                 
			if ((picture[x, y] == 1) && (2 <= B(picture, x, y, pitch)) && (B(picture, x, y, pitch) <= 6) && (A(picture, x, y, pitch) == 1) &&     //Непосредственное удаление точки, см. Zhang-Suen thinning algorithm, http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf
                        (picture[x + (y - 1)*pitch]*picture[x + 1 + y*pitch]*picture[x + (y + 1)*pitch] == 0) &&
                        (picture[x + 1 + y*pitch]*picture[x + (y + 1)*pitch]*picture[x - 1 + y*pitch] == 0))
                    {
                        picture[x + y*pitch] = 0;
                    }
			
			if ((picture[x + y*pitch] == 1) && (2 <= B(picture, x, y, pitch)) && (B(picture, x, y, pitch) <= 6) && (A(picture, x, y, pitch) == 1) &&
                        (picture[x + (y - 1)*pitch] * picture[x + 1 + y*pitch] * picture[x - 1 + y*pitch] == 0) &&
                        (picture[x * (y - 1)*pitch] * picture[x + (y + 1)*pitch] * picture[x - 1 + y*pitch] == 0))
                    {
                        picture[x + y*pitch] = 0;
                    } 

            if ((picture[x, y] == 1) &&
                (((picture[x, (y - 1)*pitch] * picture[x + 1 + y*pitch] == 1) && (picture[x - 1 + (y + 1)*pitch] != 1)) || ((picture[x + 1 + y*pitch] * picture[x + (y + 1)*pitch] == 1) && (picture[x - 1 + (y - 1)*pitch] != 1)) ||      //Небольшая модификцаия алгоритма для ещё большего утоньшения
                ((picture[x + (y + 1)*pitch] * picture[x - 1 + y*pitch] == 1) && (picture[x + 1 + (y - 1)*pitch] != 1)) || ((picture[x + (y - 1)*pitch] * picture[x - 1 + y*pitch] == 1) && (picture[x + 1 + (y + 1)*pitch] != 1))))
            {
                picture[x + y*pitch] = 0;
            }

            return picture;
}








int main()
{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };

    // Add vectors in parallel.
	int size;
	double **picture = (double**)malloc(size*size*sizeof(double*));
	for(int i = 0; i < size; i++){
		picture[i] = (double*)malloc(10*sizeof(double));
	}

	pitch ...	
    cudaError_t cudaStatus = addWithCuda(pitcure, pitch);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
	for(int i = 0; i < size; i++){
		free(picture[i]);
	}
	free(picture);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
