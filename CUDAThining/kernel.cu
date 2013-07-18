//CUDAThinning
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include "CUDAArray.cuh"

//#include<MinutiaMatching.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
cudaError_t addWithCuda(int *picture, int width, int height, int *result);

//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}

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

__device__ int B(int *picture, int x, int y, size_t pitch)        //Ìåòîä Â(Ð) âîçâðàùàåò êîëè÷åñòâî ÷åðíûõ ïèêñåëåé â îêðåñòíîñòè òî÷êè Ð
{
	int rowWidthInElements = pitch/sizeof(size_t);
	//return picture[x + (y - 1)*rowWidthInElements] + picture[x + 1 + (y - 1)*rowWidthInElements] + picture[x + 1 + y*rowWidthInElements] + picture[x + 1 + (y + 1)*rowWidthInElements] +
 //          picture[x * (y + 1)*rowWidthInElements] + picture[x - 1 + (y + 1)*rowWidthInElements] + picture[x - 1 + y*rowWidthInElements] + picture[x - 1 * (y - 1)*rowWidthInElements];

	return picture[x + (y - 1)*rowWidthInElements] + picture[x + 1 + (y - 1)*rowWidthInElements] + picture[x + 1 + y*rowWidthInElements] + picture[x + 1 + (y + 1)*rowWidthInElements] +
		   picture[x + (y + 1)*rowWidthInElements] + picture[x - 1 + (y + 1)*rowWidthInElements] + picture[x - 1 + y*rowWidthInElements] + picture[x - 1 + (y - 1)*rowWidthInElements];
			
}

__device__ int A(int *picture, int x, int y, size_t pitch)        //Ìåòîä À(Ð) âîçâðàùàåò êîëè÷åñòâî ïîäðÿä èäóùèõ áåëûõ è ÷åðíûõ ïèêñåëåé âîêðóã òî÷êè Ð (..0->1..)
{
	int rowWidthInElements = pitch/sizeof(size_t);
	int counter = 0;
    if((picture[x + (y - 1)*rowWidthInElements] == 0) && (picture[x + 1 + (y - 1)*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x + 1 + (y - 1)*rowWidthInElements] == 0) && (picture[x + 1 + y*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x + 1 + y*rowWidthInElements] == 0) && (picture[x + 1 + (y + 1)*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x + 1 + (y + 1)*rowWidthInElements] == 0) && (picture[x + (y + 1)*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x + (y + 1)*rowWidthInElements] == 0) && (picture[x - 1 + (y + 1)*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x - 1 + (y + 1)*rowWidthInElements] == 0) && (picture[x - 1 + y*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x - 1 + y*rowWidthInElements] == 0) && (picture[x - 1 + (y - 1)*rowWidthInElements] == 1))
    {
        counter++;
    }
    if ((picture[x - 1 + (y - 1)*rowWidthInElements] == 0) && (picture[x + (y - 1)*rowWidthInElements] == 1))
    {
        counter++;
    }
    return counter;
}

__global__ void compare(int* pictureToRemove, int* picture, size_t pitch, int width, int height)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
	
	if((pictureToRemove[y*rowWidthInElements + x] == 0) && (x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{
		picture[y*rowWidthInElements + x] = 0;
		pictureToRemove[y*rowWidthInElements + x] = 1;
	}

}

__global__ void ThiningImgWithCUDA(CUDAArray<int> thinnedPicture, int width, int height)
{
	int column = defaultColumn();
	int row = defaultRow();
	thinnedPicture.SetAt(row, column, 1);
	//if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	//{
	//	if ((picture[j, i] == 1) && (2 <= B(picture, j, i)) && (B(picture, j, i) <= 6) && (A(picture, j, i) == 1) &&     //Непосредственное удаление точки, см. Zhang-Suen thinning algorithm, http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf
 //                       (picture[j, i - 1]*picture[j + 1, i]*picture[j, i + 1] == 0) &&
 //                       (picture[j + 1, i]*picture[j, i + 1]*picture[j - 1, i] == 0))
 //                   {
 //                       picture[j, i] = 0;
 //                   }
	//}

}


__global__ void ThiningPictureWithCUDA(int* newPicture, int *picture ,size_t pitch, int width, int height,bool* hasChanged)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
    //if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{             
		if ((picture[x + y*rowWidthInElements] == 1) && (2 <= B(picture, x, y, pitch)) && (B(picture, x, y, pitch) <= 6) && (A(picture, x, y, pitch) == 1) &&                         
			 (picture[x + (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] == 0) &&
             (picture[x + 1 + y*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 0))
         {
				newPicture[x + y*rowWidthInElements] = 0;
                hasChanged[0] = true;
		 }	
		
	}
	//newPicture[x+rowWidthInElements*y] = picture[x+rowWidthInElements*y];
}

__global__ void ThiningPictureWithCUDA2(int* newPicture, int *picture ,size_t pitch, int width, int height, bool* hasChanged)
{
	//int *picture = newPicture;
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{             
		if ((picture[x + y*rowWidthInElements] == 1) && (2 <= B(picture, x, y, pitch)) && (B(picture, x, y, pitch) <= 6) && (A(picture, x, y, pitch) == 1) &&
			(picture[x + (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 0) &&
			(picture[x + (y - 1)*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 0))
		{
			newPicture[x + y*rowWidthInElements] = 0;
			hasChanged[0] = true;
		} 
	}
	//newPicture[x+rowWidthInElements*y] = picture[x+rowWidthInElements*y];

}

//Doesn't work correctly with parallel method
//__global__ void ThiningPictureWithCUDA3(int* newPicture, int *picture ,size_t pitch, int width, int height)
//{
//	int x = threadIdx.x + blockIdx.x*blockDim.x;
//    int y = threadIdx.y + blockIdx.y*blockDim.y;
//	int rowWidthInElements = pitch/sizeof(size_t);
//	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
//	{           
//		if ((picture[x + y*rowWidthInElements] == 1) &&
//		   (((picture[x + (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] == 1) && (picture[x - 1 + (y + 1)*rowWidthInElements] != 1)) || ((picture[x + 1 + y*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] == 1) && (picture[x - 1 + (y - 1)*rowWidthInElements] != 1)) ||      //Небольшая модификцаия алгоритма для ещё большего утоньшения
//           (( picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y - 1)*rowWidthInElements] != 1)) || ((picture[x + (y - 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y + 1)*rowWidthInElements] != 1))))
//        {
//			newPicture[x + y*rowWidthInElements] = 0;
//        }
//	}else
//	{
//		newPicture[x+rowWidthInElements*y] = picture[x+rowWidthInElements*y];
//	}
//}

void DeleteCorners(int *picture, int width, int height)
{
	
	//int x = threadIdx.x + blockIdx.x*blockDim.x;
    //int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = width;
	for(int x = 1; x < width - 1; x++)
	{
		for(int y = 1; y < height - 1; y++)
		{
			if ((picture[x + y*rowWidthInElements] == 1) &&
				(((picture[x + (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] == 1) && (picture[x - 1 + (y + 1)*rowWidthInElements] != 1)) || ((picture[x + 1 + y*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] == 1) && (picture[x - 1 + (y - 1)*rowWidthInElements] != 1)) ||      //Небольшая модификцаия алгоритма для ещё большего утоньшения
				(( picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y - 1)*rowWidthInElements] != 1)) || ((picture[x + (y - 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y + 1)*rowWidthInElements] != 1))))
			{
				picture[x + y*rowWidthInElements] = 0;
			}
		}
	}
}








int main()
{
	//int size = 32;
	int width; //= size;
	int	height; //= size;
	CUDAArray<float> img = loadImage("C:\\temp\\104_6_Binarizated.bin", true);
	width = img.Width;
	height = img.Height;
	int *picture = (int*)malloc(width*height*sizeof(int));
	int *result = (int*)malloc(width*height*sizeof(int));
	float* picture1;// = (float*)malloc(width*height*sizeof(float));
	FILE *in = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.in","r");
	FILE *out = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.out","w");
	float* result1 = (float*)malloc(width*height*sizeof(float));

	picture1 = img.cpuPt;
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			picture[j*width + i] = picture1[j*width + i] > 0 ? 0 : 1;
		}
	}

	//for(int i = 0; i < width; i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{
	//		printf("%.0f ",picture1[j*width + i]);
	//	}
	//	printf("\n");
	//}
	//printf("\n");

	//for(int i = 0; i < width; i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{
	//		fscanf(in,"%d",&picture[j*size + i]);
	//	}
	//}

	//for(int i = 0; i < width; i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{
	//		printf("%d ",picture[j*size + i]);
	//	}
	//	printf("\n");
	//}
	//printf("\n");

    cudaError_t cudaStatus = addWithCuda(picture, width, height, result); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			fprintf(out,"%d ",result[j*width + i]);
		}
		fprintf(out,"\n");
	}
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			result1[j*width + i] = result[j*width + i] > 0 ? 0 : 255;
		}
	}
	SaveArray(result1, width, height,"C:\\temp\\104_6_BinarizatedThinnedCUDA.bin");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }

	free(picture);
	free(result);
	img.Dispose();
	free(picture1);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *picture, int width, int height, int *result)
{
	int* dev_picture; 
	int* dev_pictureThinned;
	int* dev_pictureToRemove;
	bool hasChanged;
	bool* dev_hasChanged;
	int *pictureToRemove = (int*)malloc(width*height*sizeof(int));
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			pictureToRemove[j*width + i] = 1;
		}
	}
	//CUDAArray<int> img = CUDAArray<int>(picture, width, height);
	//CUDAArray<int> imgout = CUDAArray<int>(dev_pictureThinned, width, height);
//	img.
    cudaError_t cudaStatus;
	size_t pitch;
    size_t pitch1;
	size_t pitch2;
	
	cudaStatus = cudaMallocPitch((void**)&dev_pictureToRemove, &pitch2, width*sizeof(int), height);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch");
        goto Error;
    }
	cudaStatus = cudaMemcpy2D(dev_pictureToRemove, pitch2, pictureToRemove, width*sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
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
	//Allocate GPU buffers for picture.
	cudaStatus = cudaMallocPitch((void**)&dev_picture, &pitch, width*sizeof(int), height);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_hasChanged, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc!");
        goto Error;
    }

	cudaStatus = cudaMallocPitch((void**)&dev_pictureThinned, &pitch1, width*sizeof(int), height);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

    // Copy input picture from host memory to GPU buffers.

    cudaStatus = cudaMemcpy2D(dev_picture, pitch, picture, width*sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    int dimA = width*height;
    int numThreadsPerBlock = 16;
    int numBlocks = dimA / numThreadsPerBlock;
    
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

	//for(int i = 0; i < pitch1/sizeof(size_t); i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{	

	//			dev_pictureThinned[i*(pitch1/sizeof(size_t)) + j] = 1;

	//	}
	//}

	do{
		hasChanged = false;
		cudaStatus = cudaMemcpy(dev_hasChanged, &hasChanged, sizeof(bool), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		ThiningPictureWithCUDA<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(dev_pictureToRemove, dev_picture, pitch, width, height, dev_hasChanged);

		compare<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(dev_pictureToRemove, dev_picture, pitch, width, height);

		
		cudaStatus = cudaMemcpy2D(result, width*sizeof(int), dev_picture, pitch, width*sizeof(int), height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		//for(int i = 0; i < width; i++)
		//{
		//	for(int j = 0; j < height; j++)
		//	{
		//		printf("%d ",result[j*width + i]);
		//	}
		//	printf("\n");
		//}
		//printf("\n");
		
		ThiningPictureWithCUDA2<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(dev_pictureToRemove, dev_picture, pitch, width, height, dev_hasChanged);

		compare<<<dim3(ceilMod(width,16),ceilMod(height,16)),dim3(16,16)>>>(dev_pictureToRemove, dev_picture, pitch, width, height);


		cudaStatus = cudaMemcpy2D(result, width*sizeof(int), dev_picture, pitch, width*sizeof(int), height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		//for(int i = 0; i < width; i++)
		//{
		//	for(int j = 0; j < height; j++)
		//	{
		//		printf("%d ",result[j*width + i]);
		//	}
		//	printf("\n");
		//}
		//printf("\n");

		cudaStatus = cudaMemcpy(&hasChanged, dev_hasChanged, sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	}while(hasChanged);
	

	
	/*ThiningPictureWithCUDA2<<<dim3(2,2),dim3(16,16)>>>(dev_picture, dev_pictureThinned, pitch1, width, height);
	ThiningPictureWithCUDA<<<dim3(2,2),dim3(16,16)>>>(dev_picture, dev_pictureThinned, pitch1, width, height);
	ThiningPictureWithCUDA3<<<dim3(2,2),dim3(16,16)>>>(dev_pictureThinned, dev_picture, pitch, width, height);*/
	
	//ThiningImgWithCUDA<<<dim3(2,2),dim3(16,16)>>>(img, width, height);
	//result = img.GetData();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
	
	cudaStatus = cudaMemcpy2D(result, width*sizeof(int), dev_picture, pitch, width*sizeof(int), height, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	
	DeleteCorners(result, width, height);

    // Copy output vector from GPU buffer to host memory.

	//for(int i = 0; i < width; i++)
	//{
	//	for(int j = 0; j < height; j++)
	//	{
	//		printf("%d ",result[j*width + i]);
	//	}
	//	printf("\n");
	//}

Error:
    cudaFree(dev_picture);
    cudaFree(dev_pictureThinned);
	cudaFree(dev_pictureToRemove);
	cudaFree(dev_hasChanged);
	free(pictureToRemove);
	//img.Dispose();
    //cudaFree(dev_b);
    
    return cudaStatus;
}
