#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "CUDAArray.cuh"

cudaError_t addWithCuda(float border ,float *img_dst, float *img_src, int width, int height);

__global__ void addKernel(float border, float *img_dst, float *img_src, int width, int height, size_t pitch)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i < width && j < height)
	{
		//img_dst[i + j*pitch] = img_src[i + j*pitch] > border ? 255 : 0;
		if(img_src[i + j*pitch] > border) 
		{
			img_dst[i + j*pitch] = 255;
		} else {
			img_dst[i + j*pitch] = 0;
		}
	}
}

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
	sourceImage.cpuP = ar2;
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


int main()
{
    float *img_dst, *img_src;
	int width = 10;
	int height = 20;
	float border = 50;

	CUDAArray<float> img = loadImage("C:\\temp\\104_6.bin", true);
	width = img.Width;
	height = img.Height;

	img_dst = (float *) malloc(sizeof(float)*width*height);

	img_src = img.cpuP;//(float *) malloc(sizeof(float)*width*height);


	/*for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			img_src[i*width + j] = i*width + j;//i*2 + j;
		}
	}*/

    // Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(border, img_dst, img_src, width, height);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	/*
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			printf("%.0f ",img_src[i*width + j]);
		}
		printf("\n");
	}

	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			printf("%.0f ",img_dst[i*width + j]);
		}
		printf("\n");
	}*/

	SaveArray(img_dst, width, height,  "C:\\temp\\104_6_2.bin");

//	free(img_dst);
	free(img_src);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

//CUDAArray<float> img = loadImage("C:\\temp\\104_6.bin", true);
//	SaveArray(img_dst, width, height,  "C:\\temp\\104_6_1.bin");

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float border ,float *img_dst, float *img_src, int width, int height)
{
    float *dev_img_dst = 0;
    float *dev_img_src = 0;

	size_t pitch;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMallocPitch((void**)&dev_img_dst, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMallocPitch((void**)&dev_img_src, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy2D(dev_img_src, pitch, img_src, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	int numThread = 32;


    // Launch a kernel on the GPU with one thread for each element.
	addKernel<<<dim3((width + numThread - 1)/numThread,(height + numThread - 1)/numThread), dim3(numThread, numThread)>>>(border ,dev_img_dst, dev_img_src, width, height,pitch/sizeof(float));

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy2D(img_dst, width*sizeof(float), dev_img_dst, pitch, width*sizeof(float), height, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	cudaFree(dev_img_dst);
	cudaFree(dev_img_src);
    
    return cudaStatus;
}