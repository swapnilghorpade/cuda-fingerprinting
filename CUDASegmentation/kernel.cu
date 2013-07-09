#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionHelper.h"
#include <stdio.h>

#define ceilMod(x, y) (x+y-1)/y

typedef struct 
	{
		int X;
		int Y;
	} Point;

typedef struct
	{
		int AreaNumber;
		int AreaSize;
		Point* Points;
	} AreaStruct;
  			
 __device__ int defaultRow()
{
	return blockIdx.y*blockDim.y+threadIdx.y;
}

__device__ int defaultColumn()
{
	return blockIdx.x*blockDim.x+threadIdx.x;
}

 static int defaultThreadCount = 32;
 

__global__ void cudaGetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	int row = defaultRow();
	int column = defaultColumn();
	float newValue = xGradient.At(row,column)*xGradient.At(row,column) +yGradient.At(row,column)*yGradient.At(row,column);
	newValue = sqrt(newValue);
	magnitude.SetAt(row,column, newValue);
}

void GetMagnitude(CUDAArray<float> magnitude, CUDAArray<float> xGradient, CUDAArray<float> yGradient)
{
	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = 
		dim3(ceilMod(magnitude.Width,defaultThreadCount),
		ceilMod(magnitude.Height,defaultThreadCount));

	cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, xGradient, yGradient);
	cudaError_t error = cudaDeviceSynchronize();
}

__global__ void cudaGetMask(CUDAArray<float> initialArray, CUDAArray<bool> mask, int blockSize, float average)
{
	float sum = 0;
	for(int i; i<blockSize; i++)
	{
		for(int j; j<blockSize; j++)
		{
			if(defaultRow()*blockSize+j<initialArray.Height&&
				defaultColumn()*blockSize+i<initialArray.Width)
			{
			sum += initialArray.At(defaultRow()*blockSize+j,defaultColumn()*blockSize+i);
			}
		}
	}
	sum = sum/(blockSize*blockSize);
	mask.SetAt(defaultRow(),defaultColumn(),!(sum < average));
}

float GetAverageFromArray(CUDAArray<float> arrayToAverage)
{
	float sum = 0;
	float* ar = arrayToAverage.GetData();
	for(int i; i<arrayToAverage.Width; i++)
	{		
		for(int j; j<arrayToAverage.Height; j++)
		{
			sum+= ar[i+j*arrayToAverage.Width];
		}
	}
	return sum/(float)(arrayToAverage.Height*arrayToAverage.Width);
	free(ar);
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

	CUDAArray<float> sourceImage = CUDAArray<float>(ar2,width,height);

	free(ar2);		

	return sourceImage;
}

void SaveMask(CUDAArray<bool> mask, const char* name)
{
	FILE* f = fopen(name,"w");
	bool* maskOnCPU = mask.GetData();
	int* ar = (int*)malloc(sizeof(char)*(mask.Width*2+1)*mask.Height);
	int k =0;
	for(int i =0; i<mask.Height; i++)
	{
		for(int j =0; j<mask.Width; j++)
		{
			ar[k++] = (char)maskOnCPU[j+i*mask.Width]?49:48;
			ar[k++] = ' ';
		}
		ar[k++] = '\n';
	}
	fwrite(ar,sizeof(char),(mask.Width*2+1)*mask.Height,f);
	fclose(f);
}


  int main()
  {
	  //parameters
	  float weightConstant = 0.3; 
	  int windowSize = 12;
	  int threshold = 5;

	  //source image
	  CUDAArray<float> source = loadImage("C:\\temp\\104_6.bin");
	  int xSizeImg = source.Width;		  
	  int ySizeImg = source.Height;

	  // Sobel:	  
	  CUDAArray<float> xGradient = CUDAArray<float>(xSizeImg,ySizeImg);
	  CUDAArray<float> yGradient = CUDAArray<float>(xSizeImg,ySizeImg);

	  float xKernelCPU[3][3] = {{-1,0,1},
							{-2,0,2},
							{-1,0,1}};
	  CUDAArray<float> xKernel = CUDAArray<float>(*xKernelCPU,3,3);
	  
	  float yKernelCPU[3][3] = {{-1,-2,-1},
							{0,0,0},
							{1,2,1}};
	  CUDAArray<float> yKernel = CUDAArray<float>(*yKernelCPU,3,3);
	  
	  Convolve(xGradient, source, xKernel);
	  Convolve(yGradient, source, yKernel);

	  //magnitude of gradient
	  CUDAArray<float> magnitude = CUDAArray<float>(xSizeImg,ySizeImg);


	  xGradient.Dispose();
	  yGradient.Dispose();
	  xKernel.Dispose();
	  yKernel.Dispose();

	  //average magnitude 
	  float average = GetAverageFromArray(magnitude);

	  //dementions of mask
	  int N = (int)ceil(((double)source.Width) / windowSize);
	  int M = (int)ceil(((double)source.Height) / windowSize);
	  
	  //thread configuration in CUDA
	  	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
		dim3 gridSize =dim3(ceilMod(N,defaultThreadCount),
							ceilMod(M,defaultThreadCount));

		//mask creation
		CUDAArray<bool> mask = CUDAArray<bool>(N,M);
		cudaGetMask<<<gridSize, blockSize>>>(magnitude, mask, windowSize, average*weightConstant);
		
//save mask
		SaveMask(mask, "C:\\temp\\mask.txt");
		mask.Dispose();
return 0;
}

bool* PostProcessing(bool* mask, int maskX, int maskY, int threshold)
{
	AreaStruct* blackAreas = GenerateAreas(mask, maskX, maskY, true);
	mask = FillAreas(blackAreas, mask, maskX, maskY, threshold);
	AreaStruct* imageAreas = GenerateAreas(mask, maskX, maskY, false);
	mask = FillAreas(imageAreas, mask, maskX, maskY, threshold);

	return mask;
}

__global__ void changeColor(bool* dev_mask, Point* dev_toRestore, int toRestoreCounter)
{
	// coordinates of points in dev_toRestores
	int columnX = blockIdx.x*blockIdx.y*blockDim.x+threadIdx.y*blockDim.x + threadIdx.x;  
	Point point = dev_toRestore[columnX];
	
	dev_mask[point.X, point.Y] = !dev_mask[point.X, point.Y];
}

bool* FillAreas(AreaStruct* areas, bool* mask, int maskX, int maskY, int threshold)
{
	Point* toRestore = 0;
	Point* dev_toRestore = 0;
	bool* dev_mask = 0;
	int maskSize = maskX*maskY + 1;
	int toRestoreCounter = 0;
	int newRestorePoints = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)toRestore, maskSize * sizeof(Point));
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	for(int i = 0; i < maskSize; i++)
	{
		newRestorePoints = 0;

		if (areas[i].AreaSize < threshold && !IsNearBorder(areas[i].Points, areas[i].AreaSize, maskX, maskY))
        {
			while(newRestorePoints <= areas[i].AreaSize)
			{
				toRestore[toRestoreCounter++] = areas[i].Points[newRestorePoints];
			}
		}
	}

	cudaStatus = cudaMalloc((void**)&dev_toRestore, maskSize * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_mask, maskSize * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }

	 cudaStatus = cudaMemcpy(dev_toRestore, toRestore, maskSize * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       // goto Error;
    }

	cudaStatus = cudaMemcpy(dev_mask, mask, maskSize * sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       // goto Error;
    }

	changeColor<<<gridSize, bloockSize>>>(dev_mask, dev_toRestore, toRestoreCounter);

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching changeColor!\n", cudaStatus);
       // goto Error;
    }

    cudaStatus = cudaMemcpy(mask, dev_mask, maskSize * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       // goto Error;
    }

	cudaFree(toRestore);
	cudaFree(dev_toRestore);
	cudaFree(dev_mask);

	return mask;
}

__global__ void fillArea(AreaStruct* dev_areas, int areasSize, int maskX, int i, int j, bool isFirst)
{
    int columnX = defaultColumn();
	int rowY = defaultRow();
	AreaStruct area =  dev_areas[rowY*maskX + columnX];
	int iSearch = i;
	int jSearch = j-1;

	if (isFirst)
	{
		iSearch = i-1;
		jSearch = j;
	}
	
	for (int i = 0; i < areasSize; i++)
	{
		if (area.Points[i].X == iSearch && area.Points[i].Y == jSearch)
		{
			area.Points[area.AreaSize + 1].X = i;
			area.Points[area.AreaSize + 1].Y = j;
			area.AreaNumber++;
			dev_areas[columnX*maskX + rowY].Points = area.Points;
			return;
		}
	}
}

AreaStruct* GenerateAreas(bool* mask, int maskX, int maskY, bool isBlack)
{
	int areasSize = maskX * maskY + 1;
	int areaIndex = 0;
	bool isLeftImageTopBlack, isLeftBlackTopImage, isLeftBlackTopBlack;
	AreaStruct* areas = 0;
	AreaStruct* dev_areas = 0;
	Point* initialPoints = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)initialPoints, areasSize * sizeof(Point));
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	AreaStruct initialValue = MakeInitialValue(areas, initialPoints, areasSize);

	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(maskX,defaultThreadCount), ceilMod(maskY,defaultThreadCount));
	
	cudaStatus = cudaMalloc((void**)areas, areasSize * sizeof(AreaStruct));
    if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_areas, areasSize * sizeof(AreaStruct));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }
	
	for (int i = 0; i < maskX; i++)
    {
		for (int j = 0; j < maskY; j++)
        {
			if (!mask[i, j] && isBlack || mask[i, j] && !isBlack)
            {
				continue;
            }
			
			isLeftBlackTopImage = IsLeftBlackTopImage(i, j, mask[i, j - 1], mask[i - 1, j], isBlack);
			isLeftImageTopBlack = IsLeftImageTopBlack(i, j, mask[i, j - 1], mask[i - 1, j], isBlack);
			isLeftBlackTopBlack = IsLeftBlackTopBlack(i, j, mask[i, j - 1], mask[i - 1, j], isBlack);

			if (isLeftBlackTopBlack)
            {
				MergeAreas(areas, initialValue, maskX, areasSize, i, j, areaIndex);
				areaIndex--;

				continue;
            }

			if (isLeftBlackTopImage || isLeftImageTopBlack)
            {
				cudaStatus = cudaMemcpy(dev_areas, areas, areasSize * sizeof(AreaStruct), cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) 
				{
					fprintf(stderr, "cudaMemcpy failed!");
					// goto Error;
				}
				
				if (isLeftBlackTopImage)
				{
					fillArea<<<gridSize, bloockSize>>>(dev_areas, areasSize, maskX, i-1, j, i, j, true);
				}
				else
				{
					fillArea<<<gridSize, bloockSize>>>(dev_areas, areasSize, maskX, i, j-1, i, j, false);
				}

				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fillArea!\n", cudaStatus);
				   // goto Error;
				}

				cudaStatus = cudaMemcpy(areas, dev_areas, areasSize * sizeof(AreaStruct), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!");
				   // goto Error;
				}

                continue;
            }
			
			Point newPoint = {i, j};
			Point* newPoints = 0;
			cudaStatus = cudaMalloc((void**)newPoints, areasSize * sizeof(Point));
			newPoints[0] = newPoint;
			AreaStruct newArea = {areaIndex, 1, newPoints};
			areas[areaIndex++] = newArea;
		}
	}

	cudaFree(dev_areas);
	cudaFree(initialPoints);
	
	return areas;
} 

AreaStruct MakeInitialValue(AreaStruct* areas, Point* initialPoints, int areasSize)
{
	Point point = {-1,-1};
	cudaError_t cudaStatus;

	for (int i = 0; i < areasSize; i++)
	{
		initialPoints[i] = point;
	}

	AreaStruct initialValue = MakeInitialValue(); 

	for (int i = 0; i < areasSize; i++)
	{
		areas[i] = initialValue;
	}

	AreaStruct result = {-1, 0, initialPoints};

	return result;
}

bool IsLeftBlackTopBlack(int i, int j, bool topValue, bool leftValue, bool isBlack)
{
	return (j - 1 >= 0 && (topValue || isBlack) && !(topValue && isBlack) &&					//top block is black 
            i - 1 >= 0 && (leftValue || isBlack) && !(leftValue && isBlack));					//left block is black
}

bool IsLeftBlackTopImage(int i, int j, bool topValue, bool leftValue, bool isBlack) 
{
	return (i - 1 >= 0 && (leftValue || isBlack) && !(leftValue && isBlack) &&					//left block is black
           (j - 1 >= 0 && (topValue || !isBlack) && !(topValue && !isBlack)) || j - 1 < 0);	    //top block is not black or not exist
}

bool IsLeftImageTopBlack(int i, int j, bool topValue, bool leftValue, bool isBlack) 
{
	return (j - 1 >= 0 && (topValue || isBlack) && !(topValue && isBlack) &&					//top block is black 
           (i - 1 >= 0 && (leftValue || !isBlack) && !(leftValue && !isBlack)) || i - 1 < 0);	//left block is not black or not exist
}

void MergeAreas(AreaStruct* areas, AreaStruct initialValue, int maskX, int areasSize, int i, int j, int areaIndex)
{
	int areaNumberi = 0;
	int areaNumberj = 0;
	AreaStruct area = areas[j*maskX + i];

	for (int k = 0; k < areasSize; k++)
	{
		if (area.Points[k].X == i && area.Points[k].Y == j-1)
		{
			areaNumberj = k;
		}
		if (area.Points[k].X == i-1 && area.Points[k].Y == j)
		{
			areaNumberi = k;
		}
	}
                       
	if (areaNumberi != areaNumberj)
	{
		for (int k = 0; k < areas[areaNumberj].AreaSize; k++)
		{
			areas[areaNumberi].Points[areas[areaNumberi].AreaSize++] 
				= areas[areaNumberj].Points[k];
		}

		areas[areaNumberj] = initialValue;

		for (int k = areaNumberj + 1; k < areaIndex; k++)
		{
			areas[k-1] = areas[k];
		}
	}
    
	Point p = {i,j};
	areas[areaNumberi].Points[areas[areaNumberi].AreaSize++] = p;
}

bool IsNearBorder(Point* points, int size, int xBorder, int yBorder)
{
	for (int i = 0; i < size; i++)
    {
		if (points[i].X == 0 || 
			points[i].Y == 0 ||
			points[i].X == xBorder || 
			points[i].Y == yBorder)
		{
			return true;
		}
	}

	return false;
}
