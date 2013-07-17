
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define ceilMod(x, y) (x+y-1)/y


struct Minutiae
{
	int x;
	int y;
	int numMinutiaeAround;
	float angle;
};

cudaError_t addWithCuda(int* picture, int width, int height, Minutiae *result, int* minutiaeCounter, int radius);



__device__ int CheckMinutiae(int *picture, int x, int y, size_t pitch) 
{                                               
    // 1 - ending, >2 - branching,                     
    int counter = 0;
	int rowWidthInElements = pitch/sizeof(size_t);
    for (int i = x - 1; i <= x + 1; i++)
    {
        for (int j = y - 1; j <= y + 1 ; j++)
        {
            if ((picture[i + j*rowWidthInElements] == 1) && ((i != x) || (j != y))) 
			{
				counter++;
			}
        }
    }
	//return counter;
	return counter == 2?0:counter;
}

__global__  void FindMinutiae(int* picture, size_t pitch, int width, int height, Minutiae *result, int* minutiaeCounter, int* test)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)) && (picture[x + y*rowWidthInElements] == 1))
	{
		int value = CheckMinutiae(picture, x, y, pitch);
		test[x + y*rowWidthInElements] = value;
		if(value > 0)
		{
			int counter = atomicAdd(minutiaeCounter, 1);
			result[counter].x = x;
			result[counter].y = y;
		}
			
	}
	else
	{
		test[x + y*rowWidthInElements] = 0;
	};
}

__global__ void ConsiderDistanceBetweenMinutiae(int radius, Minutiae *listMinutiae, size_t pitch, int* result)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
	int dX = listMinutiae[x].x - listMinutiae[y].x; 
	int dY = listMinutiae[x].y - listMinutiae[y].y;
	if(dX*dX + dY*dY < radius*radius)
	{
		result[x + y*rowWidthInElements] = 1;
		//int counter = atomicAdd(&listMinutiae[x].numMinutiaeAround, 1);
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
	FILE *in = fopen("C:\\Users\\CUDA Fingerprinting2\\picture2.in","r");
	FILE *out = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.out","w");
	int radius = 5;
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			fscanf(in,"%d",&picture[j*width + i]);
		}
	}

	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			printf("%d ",picture[j*width + i]);
		}
		printf("\n");
	}
	printf("\n");

    cudaError_t cudaStatus = addWithCuda(picture, width, height, result, minutiaeCounter, radius); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	for(int j = 0; j < minutiaeCounter[0]; j++)
	{
		fprintf(out,"%d %d \n",result[j].x, result[j].y);
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
cudaError_t addWithCuda(int* picture, int width, int height, Minutiae *result, int* minutiaeCounter, int radius)
{
	cudaError_t cudaStatus;
	size_t pitch, pitch1, pitch2;
	int* dev_picture;
	int *dev_test;
	int* test = (int*)malloc(width*height*sizeof(int));
	int* dev_resultSpecial;
	Minutiae* dev_result;
	minutiaeCounter[0] = 0;
	int* dev_minutiaeCounter;
	int* resultSpecial;
    

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	cudaStatus = cudaMallocPitch((void**)&dev_picture, &pitch, width*sizeof(int), height);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }
	
	cudaStatus = cudaMallocPitch((void**)&dev_test, &pitch1, width*sizeof(int), height);
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

	cudaStatus = cudaMemset(dev_minutiaeCounter, 0, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }

	cudaStatus = cudaMemcpy2D(dev_picture, pitch, picture, width*sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }
	



    // Launch a kernel on the GPU with one thread for each element.
    FindMinutiae<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(dev_picture, pitch, width, height, dev_result, dev_minutiaeCounter, dev_test);

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

	for(int i = 0; i < minutiaeCounter[0]; i++)
	{
		cudaStatus = cudaMemset(&(dev_result[i]).numMinutiaeAround, 0, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy!");
			goto Error;
		}
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, dev_result, minutiaeCounter[0]*sizeof(Minutiae), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}
	
	cudaStatus = cudaMemcpy2D(test, width*sizeof(int), dev_test, pitch1, width*sizeof(int), height, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			printf("%d ",test[j*width + i]);
		}
		printf("\n");
	}

	printf("minutiaeCounter[0] = %d \n", minutiaeCounter[0]);
	for(int j = 0; j < minutiaeCounter[0]; j++)
	{
		printf("%d %d \n",result[j].x, result[j].y);
	}
//__________________________________
	cudaStatus = cudaMallocPitch((void**)&dev_resultSpecial, &pitch2, minutiaeCounter[0]*sizeof(int), minutiaeCounter[0]); //TODO : check pitches
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }
	
	int* tableDistance = (int*)calloc(minutiaeCounter[0]*minutiaeCounter[0], sizeof(int));
	cudaStatus = cudaMemcpy2D(dev_resultSpecial, pitch2, tableDistance, minutiaeCounter[0]*sizeof(int), minutiaeCounter[0]*sizeof(int), minutiaeCounter[0], cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }
	ConsiderDistanceBetweenMinutiae<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(radius, dev_result, pitch2, dev_resultSpecial);
	
	
	//resultSpecial = (int*)malloc(minutiaeCounter[0]*minutiaeCounter[0]*sizeof(int)); // final answer, later...

	Minutiae *result1 = (Minutiae*)malloc(width*height*sizeof(Minutiae));
	int countResult1 = 0;
	cudaStatus = cudaMemcpy(result, dev_result, minutiaeCounter[0]*sizeof(Minutiae), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}

	
	cudaStatus = cudaMemcpy2D(tableDistance, minutiaeCounter[0]*sizeof(int), dev_resultSpecial, pitch2, minutiaeCounter[0]*sizeof(int), minutiaeCounter[0], cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }
	
	for(int i = 0; i < minutiaeCounter[0]; i++)
	{
		for(int j = 0; j < minutiaeCounter[0]; j++)
		{
			printf("%d ",tableDistance[j*minutiaeCounter[0] + i]);
			tableDistance[j*minutiaeCounter[0] + i] == 1 ? result[i].numMinutiaeAround++ : result[i].numMinutiaeAround;
		}
		printf(" %d %d %d \n", result[i].x, result[i].y, result[i].numMinutiaeAround);
	}
	int max;
	for(int q = 0; (q < minutiaeCounter[0]) && (max != 0); q++)
	{
		int max = 0;
		int locmax = 0;
		int numLocMax;
		for(int i = 0; i < minutiaeCounter[0]; i++)
		{
			for(int j = 0; j < minutiaeCounter[0]; j++)
			{
				tableDistance[j*minutiaeCounter[0] + i] == 1 ? locmax++ : locmax;
			}
			if(locmax > max)
			{
				max = locmax;
				numLocMax = i;
			}
			locmax = 0;
		}
		int dX = 0;
		int dY = 0;
		for(int i = 0; (i < minutiaeCounter[0])&&(max != 0); i++)
		{
			if(tableDistance[numLocMax*minutiaeCounter[0] + i] == 1)
			{
				dX = dX + result[i].x;
				dY = dY + result[i].y;
			}

		}
		if(max == 0) { continue;} else{ 
		Minutiae minutiaS;
		Minutiae minutiaS1;
		minutiaS.x = (dX + max - 1)/max;
		minutiaS.y = (dY + max - 1)/max;
		dX = minutiaS.x - result[numLocMax].x;
		dY = minutiaS.y - result[numLocMax].y;
		minutiaS1 = result[numLocMax];
		int min =  dX*dX + dY*dY;
		for(int i = 0; i < minutiaeCounter[0]; i++)
		{
			if(tableDistance[i*minutiaeCounter[0] + numLocMax] == 1)
			{
				dX = minutiaS.x - result[i].x;
				dY = minutiaS.y - result[i].y;
				int locmin = dX*dX + dY*dY;
				if(min > locmin)
				{
					min = locmin;
					minutiaS1 = result[i];
				}
				//tableDistance[numLocMax*minutiaeCounter[0] + i] = 0;
				//tableDistance[i*minutiaeCounter[0] + numLocMax] = 0;
				for(int w = 0; w < minutiaeCounter[0]; w++)
				{
					tableDistance[i*minutiaeCounter[0] + w] = 0;
				}
			}
		}
		result1[countResult1] = minutiaS1;
		countResult1++;}
	}
	for(int i = 0; i < countResult1; i++)
	{
		printf(" %d %d \n", result1[i].x, result1[i].y);
	}


Error:
    cudaFree(dev_picture);
    cudaFree(dev_result);
    cudaFree(dev_minutiaeCounter);
	cudaFree(dev_test);
	cudaFree(dev_resultSpecial);
	free(test);
	free(tableDistance);
	//free(resultSpecial);


    return cudaStatus;
}
