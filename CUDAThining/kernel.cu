
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include "CUDAArray.h"

//#include<MinutiaMatching.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
cudaError_t addWithCuda(int *picture, int size, int *result);

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

	CUDAArray<float> sourceImage = CUDAArray<float>(ar2,width,height);

	free(ar2);		

	return sourceImage;
}


__device__ int B(int *picture, int x, int y, size_t pitch)        //Ìåòîä Â(Ð) âîçâðàùàåò êîëè÷åñòâî ÷åðíûõ ïèêñåëåé â îêðåñòíîñòè òî÷êè Ð
{
	return picture[x + (y - 1)*pitch] + picture[x + 1 + (y - 1)*pitch] + picture[x + 1 + y*pitch] + picture[x + 1 + (y + 1)*pitch] +
           picture[x * (y + 1)*pitch] + picture[x - 1 + (y + 1)*pitch] + picture[x - 1 + y*pitch] + picture[x - 1 * (y - 1)*pitch];
}

__device__ int A(int *picture, int x, int y, size_t pitch)        //Ìåòîä À(Ð) âîçâðàùàåò êîëè÷åñòâî ïîäðÿä èäóùèõ áåëûõ è ÷åðíûõ ïèêñåëåé âîêðóã òî÷êè Ð (..0->1..)
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


__global__ void ThiningPictureWithCUDA(int* newPicture, int *picture ,size_t pitch, int width, int height)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
    //if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{             
		if ((picture[x + y*rowWidthInElements] == 1) && (2 <= B(picture, x, y, rowWidthInElements)) && (B(picture, x, y, rowWidthInElements) <= 6) && (A(picture, x, y, rowWidthInElements) == 1) &&     //Íåïîñðåäñòâåííîå óäàëåíèå òî÷êè, ñì. Zhang-Suen thinning algorithm, http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf
            (picture[x + (y - 1)*rowWidthInElements]*picture[x + 1 + y*rowWidthInElements]*picture[x + (y + 1)*rowWidthInElements] == 0) &&
            (picture[x + 1 + y*rowWidthInElements]*picture[x + (y + 1)*rowWidthInElements]*picture[x - 1 + y*rowWidthInElements] == 0))
        {
            picture[x + y*rowWidthInElements] = 0;
			//thinnedPicture[x + y*rowWidthInElements] = 0;
        }
		
		//if ((picture[x + y*pitch] == 1) && (2 <= B(picture, x, y, pitch)) && (B(picture, x, y, pitch) <= 6) && (A(picture, x, y, pitch) == 1) &&
		//	(picture[x + (y - 1)*pitch] * picture[x + 1 + y*pitch] * picture[x - 1 + y*pitch] == 0) &&
		//	(picture[x * (y - 1)*pitch] * picture[x + (y + 1)*pitch] * picture[x - 1 + y*pitch] == 0))
		//{
		//	picture[x + y*pitch] = 0;
		//	//thinnedPicture[x + y*pitch] = 0;
		//} 
	}
	newPicture[x+rowWidthInElements*y] = picture[x+rowWidthInElements*y];
}

__global__ void ThiningPictureWithCUDA2(int* newPicture, int *picture ,size_t pitch, int width, int height)
{
	//int *picture = newPicture;
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
    //if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{             
		//if ((picture[x + y*pitch] == 1) && (2 <= B(picture, x, y, pitch)) && (B(picture, x, y, pitch) <= 6) && (A(picture, x, y, pitch) == 1) &&     //Íåïîñðåäñòâåííîå óäàëåíèå òî÷êè, ñì. Zhang-Suen thinning algorithm, http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf
  //          (picture[x + (y - 1)*pitch]*picture[x + 1 + y*pitch]*picture[x + (y + 1)*pitch] == 0) &&
  //          (picture[x + 1 + y*pitch]*picture[x + (y + 1)*pitch]*picture[x - 1 + y*pitch] == 0))
  //      {
  //          picture[x + y*pitch] = 0;
		//	//thinnedPicture[x + y*pitch] = 0;
  //      }
		
		if ((picture[x + y*rowWidthInElements] == 1) && (2 <= B(picture, x, y, rowWidthInElements)) && (B(picture, x, y, rowWidthInElements) <= 6) && (A(picture, x, y, rowWidthInElements) == 1) &&
			(picture[x + (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 0) &&
			(picture[x * (y - 1)*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 0))
		{
			picture[x + y*rowWidthInElements] = 0;
		} 
	}
	newPicture[x+rowWidthInElements*y] = picture[x+rowWidthInElements*y];

}


__global__ void ThiningPictureWithCUDA3(int* newPicture, int *picture ,size_t pitch, int width, int height)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
	int rowWidthInElements = pitch/sizeof(size_t);
    //if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	if((x > 0) && (y > 0) && (x < (width - 1)) && (y < (height - 1)))
	{           
		//if ((picture[x + y*rowWidthInElements] == 1) &&
  //          (((picture[x, (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] == 1) && (picture[x - 1 + (y + 1)*rowWidthInElements] != 1)) || ((picture[x + 1 + y*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] == 1) && (picture[x - 1 + (y - 1)*rowWidthInElements] != 1)) ||      //Íåáîëüøàÿ ìîäèôèêöàèÿ àëãîðèòìà äëÿ åù¸ áîëüøåãî óòîíüøåíèÿ
  //          ((picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y - 1)*rowWidthInElements] != 1)) || ((picture[x + (y - 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y + 1)*rowWidthInElements] != 1))))
  //      {
  //          picture[x + y*rowWidthInElements] = 0;
  //      }


		if ((picture[x + y*rowWidthInElements] == 1) &&
		   (((picture[x + (y - 1)*rowWidthInElements] * picture[x + 1 + y*rowWidthInElements] == 1) && (picture[x - 1 + (y + 1)*rowWidthInElements] != 1)) || ((picture[x + 1 + y*rowWidthInElements] * picture[x + (y + 1)*rowWidthInElements] == 1) && (picture[x - 1 + (y - 1)*rowWidthInElements] != 1)) ||      //Небольшая модификцаия алгоритма для ещё большего утоньшения
           (( picture[x + (y + 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y - 1)*rowWidthInElements] != 1)) || ((picture[x + (y - 1)*rowWidthInElements] * picture[x - 1 + y*rowWidthInElements] == 1) && (picture[x + 1 + (y + 1)*rowWidthInElements] != 1))))
        {
			picture[x + y*rowWidthInElements] = 0;
        }


	}
	newPicture[x+rowWidthInElements*y] = picture[x+rowWidthInElements*y];
}









int main()
{
	int size = 32;
	int *picture = (int*)malloc(size*size*sizeof(int));

	int *result = (int*)malloc(size*size*sizeof(int));
	FILE *in = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.in","r");
	FILE *out = fopen("C:\\Users\\CUDA Fingerprinting2\\picture.out","w");
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			fscanf(in,"%d",&picture[i*size + j]);
		}
	}

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			printf("%d ",picture[i*size + j]);
		}
		printf("\n");
	}

    cudaError_t cudaStatus = addWithCuda(picture, size, result); // todo: width, height
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			fprintf(out,"%d ",result[i*size + j]);
		}
		fprintf(out,"\n");
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

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *picture, int size, int *result)
{
	int* dev_picture;
	int* dev_pictureThinned;
	int width, height;
	width = size;
	height = size;

	//CUDAArray<int> img = CUDAArray<int>(picture, width, height);
	//CUDAArray<int> imgout = CUDAArray<int>(dev_pictureThinned, width, height);
//	img.
    cudaError_t cudaStatus;
	size_t pitch;
    size_t pitch1;
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
	
	cudaStatus = cudaMallocPitch((void**)&dev_pictureThinned, &pitch1, width*sizeof(int), height);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch!");
        goto Error;
    }

    // Copy input picture from host memory to GPU buffers.

    cudaStatus = cudaMemcpy2D(dev_picture, pitch, picture, width*sizeof(int), width*sizeof(int), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    int dimA = size*size;
    int numThreadsPerBlock = 16;
    int numBlocks = dimA / numThreadsPerBlock;
    
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
	ThiningPictureWithCUDA3<<<dim3(ceilMod(size,16),ceilMod(size,16)),dim3(16,16)>>>(dev_pictureThinned, dev_picture, pitch, width, height);
    //ThiningPictureWithCUDA2<<<dim3(ceilMod(size,16),ceilMod(size,16)),dim3(16,16)>>>(dev_pictureThinned, dev_picture, pitch, width, height);
	//ThiningPictureWithCUDA3<<<dim3(ceilMod(size,16),ceilMod(size,16)),dim3(16,16)>>>(dev_pictureThinned, dev_picture, pitch, width, height);
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
	

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy2D(result, width*sizeof(int), dev_picture, pitch, width*sizeof(int), height, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			printf("%d ",result[i*size + j]);
		}
		printf("\n");
	}

Error:
    cudaFree(dev_picture);
    cudaFree(dev_pictureThinned);
	//img.Dispose();
    //cudaFree(dev_b);
    
    return cudaStatus;
}
