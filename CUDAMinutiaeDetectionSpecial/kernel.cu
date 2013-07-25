//CUDAMinutiaeDetectionSpecial
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "CUDAArray.cuh"

#define ceilMod(x, y) (x+y-1)/y


struct Minutiae
{
	int x;
	int y;
	float angle;
};

void FindBigMinutiaeCUDA(int* picture, int width, int height, Minutiae *result, int* minutiaeCounter, int radius);



CUDAArray<float> loadImage(const char* name, bool sourceIsFloat = false)
{
	FILE* f = fopen(name,"rb");
			
	int width;
	int height;
	
	fread(&width,sizeof(int),1,f);
			
	fread(&height,sizeof(int),1,f);
	
	float* ar2 = (float*)malloc(sizeof(float)*width*height);

	if(!sourceIsFloat)
	{
		int* ar = (int*)malloc(sizeof(int)*width*height);
		fread(ar,sizeof(int),width*height,f);
		for(int i=0;i<width*height;i++)
		{
			ar2[i]=ar[i];
		}
		
		free(ar);
	}
	else
	{
		fread(ar2,sizeof(float),width*height,f);
	}
	
	fclose(f);

	CUDAArray<float> sourceImage = CUDAArray<float>();
	sourceImage.cpuPt = ar2;
	sourceImage.Width = width;
	sourceImage.Height = height;

	//free(ar2);		

	return sourceImage;
	//return ar2;
}

void SaveArray(float* arTest, int width, int height, const char* fname)
{
	FILE* f = fopen(fname,"wb");
	fwrite(&width,sizeof(int),1,f);
	fwrite(&height,sizeof(int),1,f);
	for(int i=0;i<width*height;i++)
	{
		float value = (float)arTest[i];
		int result = fwrite(&value,sizeof(float),1,f);
		result++;
	}
	fclose(f);
	free(arTest);
}



__device__ int CheckMinutiae(int *picture, int x, int y, size_t pitch) 
{                                               
    // 1 - ending, >2 - branching,                     
    int counter = 0;
	int rowWidthInElements = pitch/sizeof(size_t);
    for (int i = x - 1; i <= x + 1; i++)
    {
        for (int j = y - 1; j <= y + 1 ; j++)
        {
            if ((picture[i + j*rowWidthInElements] == 0) && ((i != x) || (j != y))) 
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
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)) && (picture[x + y*rowWidthInElements] == 0))
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
	//result[x + y*rowWidthInElements] = 1;
	if(dX*dX + dY*dY < radius*radius)
	{
		result[x + y*rowWidthInElements] = 1;
		//int counter = atomicAdd(&listMinutiae[x].numMinutiaeAround, 1);
	}
}


int main()
{
	int size = 32;
	int width; //= size;
	int	height;// = size;
	CUDAArray<float> img = loadImage("C:\\temp\\Thinned81_81.bin", true);
	width = img.Width;
	height = img.Height;
	int *picture = (int*)malloc(width*height*sizeof(int));
	int *minutiaeCounter = (int*)malloc(sizeof(int));
	Minutiae *result = (Minutiae*)malloc(width*height*sizeof(Minutiae));
	FILE *in = fopen("C:\\Users\\CUDA Fingerprinting2\\picture2.in","r");
	FILE *out = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.out","w");
	int radius = 5;

	float* picture1;
	picture1 = img.cpuPt;
	int* result1 = (int*)malloc(width*height*sizeof(int));
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			picture[j*width + i] = (int)picture1[j*width + i];
			result1[j*width + i] = (int)picture1[j*width + i];
		}
	}

	//for(int i = 0; i < width; i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{
	//		fscanf(in,"%d",&picture[j*width + i]);
	//	}
	//}

	FindBigMinutiaeCUDA(picture, width, height, result, minutiaeCounter, radius); 

	//fprintf(out,"%d \n", minutiaeCounter[0]);
	//for(int j = 0; j < minutiaeCounter[0]; j++)
	//{
	//	fprintf(out,"%d %d \n",result[j].x, result[j].y);
	//}
		
	for(int j = 0; j < minutiaeCounter[0]; j++)
	{
		picture1[result[j].y*width + result[j].x] = 150;
	}
	SaveArray(picture1, width, height,"C:\\temp\\MinutiaeMatched81_81.bin");


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
	free(result1);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void FindBigMinutiaeCUDA(int* picture, int width, int height, Minutiae *result, int* minutiaeCounter, int radius)
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
	//for(int i = 0; i < width; i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{
	//		printf("%d ",test[j*width + i]);
	//	}
	//	printf("\n");
	//}

	//printf("minutiaeCounter[0] = %d \n", minutiaeCounter[0]);
	//for(int j = 0; j < minutiaeCounter[0]; j++)
	//{
	//	printf("%d %d \n",result[j].x, result[j].y);
	//}
//__________________________________

	int *minutiaeCounter1 = (int*)malloc(sizeof(int));
	minutiaeCounter1[0] = minutiaeCounter[0];
	cudaStatus = cudaMallocPitch((void**)&dev_resultSpecial, &pitch2, minutiaeCounter1[0]*sizeof(int), minutiaeCounter1[0]); //TODO : check pitches
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }
	
	int* tableDistance = (int*)calloc(minutiaeCounter1[0]*minutiaeCounter1[0], sizeof(int));
	//int* tableDistance = (int*)malloc(minutiaeCounter1[0]*minutiaeCounter1[0]*sizeof(int));
	//for(int i = 0; i < minutiaeCounter1[0]; i++)
	//{
	//	for(int j = 0; j < minutiaeCounter1[0]; j++)
	//	{
	//		tableDistance[j*minutiaeCounter1[0] + i ] = 0;
	//
	//	}
	//}
	cudaStatus = cudaMemcpy2D(dev_resultSpecial, pitch2, tableDistance, minutiaeCounter1[0]*sizeof(int), minutiaeCounter1[0]*sizeof(int), minutiaeCounter1[0], cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }
	ConsiderDistanceBetweenMinutiae<<<dim3(ceilMod(minutiaeCounter1[0],16),ceilMod(minutiaeCounter1[0],16)),dim3(16,16)>>>(radius, dev_result, pitch2, dev_resultSpecial);
	
	
	//resultSpecial = (int*)malloc(minutiaeCounter1[0]*minutiaeCounter1[0]*sizeof(int)); // final answer, later...

	Minutiae *result1 = (Minutiae*)malloc(width*height*sizeof(Minutiae));
	int countResult1 = 0;
	//cudaStatus = cudaMemcpy(result, dev_result, minutiaeCounter1[0]*sizeof(Minutiae), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "cudaMemcpy failed!");
	//    goto Error;
	//}

	
	cudaStatus = cudaMemcpy2D(tableDistance, minutiaeCounter1[0]*sizeof(int), dev_resultSpecial, pitch2, minutiaeCounter1[0]*sizeof(int), minutiaeCounter1[0], cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }
	
	//for(int i = 0; i < minutiaeCounter1[0]; i++)
	//{
	//	for(int j = 0; j < minutiaeCounter1[0]; j++)
	//	{
	//		printf("%d ",tableDistance[j*minutiaeCounter1[0] + i]);
	//		//tableDistance[j*minutiaeCounter1[0] + i] == 1 ? result[i].numMinutiaeAround++ : result[i].numMinutiaeAround;
	//	}
	//	printf(" %d %d %d \n", result[i].x, result[i].y, result[i].numMinutiaeAround);
	//}
	int max = 1;
	for(int q = 0; (q < minutiaeCounter1[0]) && (max != 0); q++)
	{
		max = 0;
		int locmax = 0;
		int numLocMax;
		for(int i = 0; i < minutiaeCounter1[0]; i++)
		{
			for(int j = 0; j < minutiaeCounter1[0]; j++)
			{
				tableDistance[j*minutiaeCounter1[0] + i] == 1 ? locmax++ : locmax;
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
		for(int i = 0; (i < minutiaeCounter1[0])&&(max != 0); i++)
		{
			if(tableDistance[numLocMax*minutiaeCounter1[0] + i] == 1)
			{
				dX = dX + result[i].x;
				dY = dY + result[i].y;
			}

		}
		if(max == 0)
		{
			continue;
		} else
		{
			Minutiae minutiaS;
			Minutiae minutiaS1;
			minutiaS.x = (dX + max - 1)/max;
			minutiaS.y = (dY + max - 1)/max;
			dX = minutiaS.x - result[numLocMax].x;
			dY = minutiaS.y - result[numLocMax].y;
			minutiaS1 = result[numLocMax];
			int min =  dX*dX + dY*dY;
			for(int i = 0; i < minutiaeCounter1[0]; i++)
			{
				if(tableDistance[i*minutiaeCounter1[0] + numLocMax] == 1)
				{
					dX = minutiaS.x - result[i].x;
					dY = minutiaS.y - result[i].y;
					int locmin = dX*dX + dY*dY;
					if(min > locmin)
					{
						min = locmin;
						minutiaS1 = result[i];
					}
					//tableDistance[numLocMax*minutiaeCounter1[0] + i] = 0;
					//tableDistance[i*minutiaeCounter1[0] + numLocMax] = 0;
					for(int w = 0; w < minutiaeCounter1[0]; w++)
					{
						tableDistance[i*minutiaeCounter1[0] + w] = 0;
					}
				}
			}
			result1[countResult1] = minutiaS1;
			countResult1++;
		}
	}
	//printf("mintutiaeCounter[0] = %d\n", countResult1);
	for(int i = 0; i < countResult1; i++)
	{
		//printf(" %d %d \n", result1[i].x, result1[i].y);
		result[i] = result1[i];
	}
	result = result1;
	minutiaeCounter[0] = countResult1;


	//printf("minutiaeCounter[0] = %d \n", minutiaeCounter[0]);
	//for(int j = 0; j < minutiaeCounter[0]; j++)
	//{
	//	printf("%d %d \n",result[j].x, result[j].y);
	//}

Error:
    cudaFree(dev_picture);
    cudaFree(dev_result);
    cudaFree(dev_minutiaeCounter);
	cudaFree(dev_test);
	cudaFree(dev_resultSpecial);
	free(test);
	free(tableDistance);
	free(minutiaeCounter1);
	//free(resultSpecial);

}
