
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define ceilMod(x, y) (x+y-1)/y


struct Minutiae{
	int x;
	int y;
	bool belongToBig;
};

cudaError_t addWithCuda(int* picture, int width, int height, Minutiae *result, int* minutiaeCounter);



__device__ int CheckMinutiae(int *picture, int x, int y, size_t pitch) 
    {                                               
        int result; // 1 - ending, >2 - branching,                     
        int counter = 0;
		int rowWidthInElements = pitch/sizeof(size_t);
        for (int i = x - 1; i <= x + 1; i++)
        {
            for (int j = y - 1; j <= y + 1 ; j++)
            {
                if ((picture[i + j*rowWidthInElements] == 0) && (i != x) && (j != y)) 
				{
					counter++;
				}
            }
        }
        if (counter == 1)
        {
            return result = 1;
        }
        else
        {
            if (counter > 2)
            {
                return result = counter;
            }
            else
            {
                return result = 0;
            }
        }
    }

__global__  void FindMinutiae(int* picture, size_t pitch, int width, int height, Minutiae *result, int* minutiaeCounter)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{
		if(CheckMinutiae(picture, x, y, pitch) > 0)
		{
			Minutiae newMinutia;
			newMinutia.x = x;
			newMinutia.y = y;
			result[minutiaeCounter[0]] = newMinutia;
			minutiaeCounter[0]++;
		}
	}
}





int main()
{
	int size = 32;
	int width = size;
	int	height = size;
	int *picture = (int*)malloc(width*height*sizeof(int));
	int *minutiaeCounter = (int*)malloc(sizeof(int));
	Minutiae *result = (Minutiae*)malloc(width*height*sizeof(Minutiae));
	FILE *in = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.in","r");
	FILE *out = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.out","w");
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			fscanf(in,"%d",&picture[j*size + i]);
		}
	}

	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			printf("%d ",picture[j*size + i]);
		}
		printf("\n");
	}
	printf("\n");

    cudaError_t cudaStatus = addWithCuda(picture, width, height, result, minutiaeCounter); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	for(int j = 0; j < minutiaeCounter[0]; j++)
	{
			fprintf(out,"%d %d ",result[j].x, result[j]);
	}


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }

	free(picture);
	free(result);
	free(minutiaeCounter);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* picture, int width, int height, Minutiae *result, int* minutiaeCounter)
{
	cudaError_t cudaStatus;
	size_t pitch;
	int* dev_picture;
	Minutiae* dev_result;
	minutiaeCounter[0] = 0;
	int* dev_minutiaeCounter;
    

	cudaStatus = cudaMallocPitch((void**)&dev_picture, &pitch, width*sizeof(int), height);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }
	
	cudaStatus = cudaMalloc((void**)&dev_result,width*height*sizeof(Minutiae));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_minutiaeCounter, sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_minutiaeCounter, &minutiaeCounter, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }

	cudaStatus = cudaMemcpy2D(dev_picture, pitch, picture, width*sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }
	
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    FindMinutiae<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(dev_picture, pitch, width, height, dev_result, dev_minutiaeCounter);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	cudaStatus = cudaMemcpy(minutiaeCounter, dev_minutiaeCounter, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, dev_result, minutiaeCounter[0]*sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_picture);
    cudaFree(dev_result);
    cudaFree(dev_minutiaeCounter);

    return cudaStatus;
}
