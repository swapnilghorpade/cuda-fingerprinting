#define __CUDACC__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_types.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "math.h"
#include "math_functions.h"
#include "math_constants.h"
#include "device_functions.h"
//#include "Point.h"
extern "C"{

__declspec(dllexport) int main();

__declspec(dllexport) int initDevice();

__declspec(dllexport) int disposeDevice();

__declspec(dllexport) int createGaborFilters(float*, float, int);

__declspec(dllexport) void createFingerCode(int* imgBytes, float* result, int width, int height, 
													int filterAmount, int numBands, int numSectors,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY);

__declspec(dllexport) void fullFilterImage(int* imgBytes, int* result, int width, int height, 
													int filterAmount, int numBands,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY);

__declspec(dllexport) void normalizeImage(int* imgBytes, int* result, int width, int height, int numBands,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY);

__declspec(dllexport) void findCorePoint(int* imgBytes, int width, int height, int* xCore, int* yCore);

__declspec(dllexport) void matchFingercode(float* fingercode, float numberOfRegions, float* result);

__declspec(dllexport) void loadNoncoalescedFingerCodeDatabase(float* fingers, int numberOfRegions, int numberOfCodes);

__declspec(dllexport) void loadFingerCodeDatabase(float* fingers, int numberOfRegions, int numberOfCodes);

__declspec(dllexport) void sortArrayAndIndexes(float* arr, int* indexArray, int amount);

__declspec(dllexport) void BuildHull(int *arr, int N,int *IntHull,int *NHull);

__declspec(dllexport) void FieldFilling(bool *field,int rows, int columns,int *arr, int N);

__declspec(dllexport) void BuildWorkingArea(bool *field,int rows,int columns,int radius);

}

__global__ void bitonicSort(float* arr, int* indexArr, int amount, int comparatorBase, int comparatorShift)
{
	int itemIndex = blockIdx.x*1024+threadIdx.y*32+threadIdx.x;
	int startIndex = (comparatorShift<<1)*(itemIndex/comparatorShift)
		+itemIndex%comparatorShift;
	int endIndex = startIndex+comparatorShift;
	bool ascending =  (startIndex/comparatorBase)%2 != 0;
	float temp = arr[startIndex];
	float temp2 = arr[endIndex];
	if (ascending && temp > temp2 || !ascending && temp < temp2)
	{
		float tempIndex = indexArr[startIndex];
		float tempIndex2 = indexArr[endIndex];
		arr[startIndex] = temp2;
		arr[endIndex] = temp;
		indexArr[startIndex] = tempIndex2;
		indexArr[endIndex] = tempIndex;
	}
}

void sortArrayAndIndexes(float* arr, int* indexArray, int amount)
{
	int power = (int)(log((double)amount)/log(2.0));
	int size = (int)pow(2.0,power+1);

	float* cudaArr;
	int* cudaIndexArr;

	cudaError_t error = cudaMalloc(&cudaArr,sizeof(float)*size);
	error = cudaMalloc(&cudaIndexArr,sizeof(int)*size);

	error = cudaMemset(cudaArr,0,sizeof(float)*size);
	error = cudaMemset(cudaIndexArr,0,sizeof(float)*size);
	
	error = cudaMemcpy(cudaArr,arr,sizeof(float)*amount,cudaMemcpyHostToDevice);
	error = cudaMemcpy(cudaIndexArr,indexArray,sizeof(int)*amount,cudaMemcpyHostToDevice);
	
	for (int baseI = 2; baseI <= size; baseI *= 2)
	{
		for (int i = baseI; i >= 2; i >>= 1)
		{
			bitonicSort<<<1+size/2/1024,dim3(32,32)>>>(cudaArr, cudaIndexArr, size, baseI, i /2);               
			//error = cudaMemcpy(arr,cudaArr,sizeof(float)*amount,cudaMemcpyDeviceToHost);
			//error = cudaMemcpy(indexArray,cudaIndexArr,sizeof(int)*amount,cudaMemcpyDeviceToHost);
		}
	}
	error = cudaMemcpy(arr,cudaArr,sizeof(float)*amount,cudaMemcpyDeviceToHost);
	error = cudaMemcpy(indexArray,cudaIndexArr,sizeof(int)*amount,cudaMemcpyDeviceToHost);
	cudaFree(&cudaArr);
	cudaFree(&cudaIndexArr);
}


float* cudaFilterPointer = 0;

float* database;

int databaseCount =0;

//applies 3x3 smooth filter centered at the specified pixel
__device__ float smoothOrientationField(
	float* field, 
	int pitch,
	int xDim, 
	int yDim,
	int x,
	int y)
{
	float mean=0;
	float count=0;
	for(int i=-1;i<2;i++)
	{
		int x1 =x+i;
		if(x1<0||x1>=xDim)continue;
		for(int j=-1;j<2;j++)
		{
			int y1 = y+j;
			if(y1<0||y1>=yDim)continue;
			int index = y1*pitch+x1;
			count++;
			mean+=field[index];
		}
	}
	return mean/count;
}

// smoothes the orientation field components and returns it as an array of angles
__global__ void	makeSmoothedOrientationField(
		float* orFieldPointer, 
		float* xPointer, 
		float* yPointer,
		int pitch,
		int orFieldXDim, 
		int orFieldYDim)
{
	int x = threadIdx.x+32*blockIdx.x;
	int y = threadIdx.y+32*blockIdx.y;
	if(x<orFieldXDim&&y<orFieldYDim)
	{
		float xComp = smoothOrientationField(yPointer,pitch/sizeof(float),orFieldXDim,orFieldYDim,x,y);
		float yComp = smoothOrientationField(xPointer,pitch/sizeof(float),orFieldXDim,orFieldYDim,x,y);
		float result = 0.0f;
		if (__isnanf(xComp) ||__isnanf(yComp)) result = xComp+yComp;
		else if (!(xComp == 0.0f && yComp == 0.0f))
		{
			if (xComp > 0 && yComp >= 0)
				result = atan(yComp/xComp);
			if (xComp > 0 && yComp < 0)
				result = atan(yComp/xComp) + 2.0f*CUDART_PI_F;
			if (xComp < 0)
				result = atan(yComp/xComp) + CUDART_PI_F;
			if (xComp == 0 && yComp > 0)
				result = CUDART_PI_F/2.0f;
			if (xComp == 0 && yComp < 0)
				result = 3.0f*CUDART_PI_F/2.0f;
			result = result/2.0f + CUDART_PI_F/2.0f;
			if (result> CUDART_PI_F) result -= CUDART_PI_F;
		}
		orFieldPointer[x+y*pitch/sizeof(float)] = result;
	}
}

// matching function that calculates the score between the input and database FCode
__global__ void match(
		float* fingerCode,
		float* database,
		int numberOfRegions,
		int numberOfSamples,
		float* result)
{
	int index = blockIdx.x*32*32+threadIdx.y*32+threadIdx.x;
	int databaseIndex = index/16*numberOfRegions+threadIdx.x%16;
	if(index<numberOfSamples)
	{
		float accum =0.0f;
		for(int i=0;i<numberOfRegions;i++)
		{
			float diff = database[index+16*i]-fingerCode[i];
			accum+=diff*diff;
		}
		accum = sqrt(accum);
		result[index] = accum;
	}
}

// calculates curvature metric of the selected point. Main point of changes between different methods
// see the article for details
__global__ void calculateCoreMetric(
		float* orField, 
		float* coreMetric, 
		int pitch,
		int orFieldXDim,
		int orFieldYDim)
{
	int radius = 5;
	int squaredRadius = radius*radius;
	int x = threadIdx.x+32*blockIdx.x;
	int y = threadIdx.y+32*blockIdx.y;
	float metric = 0.0f;
	if(x<orFieldXDim&&y<orFieldYDim)
	{
		for (int i = -radius; i <= radius; i++)
		{
			if (x + i < 0 || x + i >= orFieldXDim)
				continue;
			for (int j = -radius; j <= radius; j++)
			{
				if (y + j < 0 || y + j >= orFieldYDim)
					continue;
				if (j*j + i*i > squaredRadius || j == 0 && i == 0)
					continue;
				float pointAngle = orField[x + i+pitch*( y + j)];
				// if (double.IsNaN(pointAngle)) continue; // todo: reconsider
				if (pointAngle > CUDART_PI_F*2.0f) pointAngle -= CUDART_PI_F*2.0f;
				if (pointAngle < 0) pointAngle += CUDART_PI_F*2.0f;
				float baseAngle = atan2((float)j, (float)i);
				baseAngle -= CUDART_PI_F/2;
				if (baseAngle < 0) baseAngle += CUDART_PI_F*2.0f;
				if (baseAngle > CUDART_PI_F*2.0f) baseAngle -= CUDART_PI_F*2.0f;
				float diffAngle = pointAngle - baseAngle;
				metric += abs(cos(diffAngle));
			}
		}
		coreMetric[x+pitch*y] = metric;
	}
}

// helper function for border cases in Sobel filter
__device__ int checkBoundsForSobel(int* image, int x, int y, int dx, int dy, int width, int height)
{
	if(x+dx<0||x+dx>=width||y+dy<0||y+dy>=height)return 0;
	return image[x+dx+width*(y+dy)];
}

// applies x and y Sobel filters to the image
__global__ void applySobel(int* image, int* result1,int* result2, int width, int height)
{
	int x0 = threadIdx.x+blockIdx.x*32;
	int y0 = threadIdx.y+blockIdx.y*32;
	int index = x0+y0*width;

	result1[index] = 
		checkBoundsForSobel(image,x0,y0,-1,-1,width,height)
		+2*checkBoundsForSobel(image,x0,y0,0,-1,width,height)
		+checkBoundsForSobel(image,x0,y0,1,-1,width,height)
		-checkBoundsForSobel(image,x0,y0,-1,1,width,height)
		-2*checkBoundsForSobel(image,x0,y0,0,1,width,height)
		-checkBoundsForSobel(image,x0,y0,1,1,width,height);
		
	result2[index] = 
		checkBoundsForSobel(image,x0,y0,-1,-1,width,height)
		+2*checkBoundsForSobel(image,x0,y0,-1,0,width,height)
		+checkBoundsForSobel(image,x0,y0,-1,1,width,height)
		-checkBoundsForSobel(image,x0,y0,1,-1,width,height)
		-2*checkBoundsForSobel(image,x0,y0,1,0,width,height)
		-checkBoundsForSobel(image,x0,y0,1,1,width,height);	
}

// normalizes the image to the chosen values of mean and variance
__global__ void applyNormalization(
	int* image,
	float baseMean,
	float baseVariance, // target mean and variance
	float meanIn,
	float varianceIn, // mean and variance inside the bands
	float meanOut, 
	float varianceOut, // mean and variance outside the bands
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadiusSquared, // inner band radius
	int bandMaxRadiusSquared,  // outer band radius)
	int horizontalSize)
{
	int x = blockIdx.x*32+threadIdx.x;
	int y = blockIdx.y*32+threadIdx.y;

	float value = (float)image[x+y*horizontalSize*32];
    int result = baseMean;
	int dX = x-referenceX;
	int dY = y-referenceY;

	int distance = dX*dX+dY*dY;

	if(distance<bandMaxRadiusSquared)
	{
		if (varianceIn > 0)
        {
			result +=
				((value > meanIn) ? (1) : (-1))*
				(int) sqrt(baseVariance/varianceIn*(value - meanIn)*(value - meanIn));
            if (result < 0) result = 0;
			if (result > 255.0f) result = 255;
        }
	}
	else
	{
		if (varianceOut > 0)
        {
			result +=
				((value > meanOut) ? (1) : (-1))*
				(int) sqrt(baseVariance/varianceOut*(value - meanOut)*(value - meanOut));
            if (result < 0) result = 0;
			if (result > 255.0f) result = 255;
        }
	}

	image[x+y*horizontalSize*32] = (int)result;
}

// calculates inside and outside variances for FCode area
__global__ void countVariances(
	int* image,
	float* variances,
	float meanIn,
	float meanOut,
	int squareSize,
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadiusSquared, // inner band radius
	int bandMaxRadiusSquared,  // outer band radius
	int horizontalSize,
	int verticalSize)
{
	int index = threadIdx.x+threadIdx.y*horizontalSize;
	
	int baseX = threadIdx.x*32;
	int baseY = threadIdx.y*32;

	float varianceIn =0;
	float varianceOut =0;

	for(int x=baseX;x<baseX+32;x++)
	{
		for(int y=baseY;y<baseY+32;y++)
		{
			int dX = x-referenceX;
			int dY = y-referenceY;

			int distance = dX*dX+dY*dY;
			float color = (float)image[x+y*horizontalSize*32];

			if(distance>=bandMinRadiusSquared&& distance<bandMaxRadiusSquared)
			{
				varianceIn+= (color-meanIn)*(color-meanIn);
			}
			else
			{
				varianceOut+=(color-meanOut)*(color-meanOut);
			}
		}
	}

	int outOffset = horizontalSize*verticalSize;
	variances[index] = varianceIn;
	variances[index+outOffset]=varianceOut;
}

// calculates inside and outside mean values for FCode area
__global__ void countMeans(
	int* image,
	float* means,
	int* meanCounts,
	int squareSize,
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadiusSquared, // inner band radius
	int bandMaxRadiusSquared,  // outer band radius
	int horizontalSize,
	int verticalSize)
{
	int index = threadIdx.x+threadIdx.y*horizontalSize;
	
	int baseX = threadIdx.x*32;
	int baseY = threadIdx.y*32;

	float meanIn =0;
	float meanOut =0;
	float meanCountIn =0;
	float meanCountOut =0;

	for(int x=baseX;x<baseX+32;x++)
	{
		for(int y=baseY;y<baseY+32;y++)
		{
			int dX = x-referenceX;
			int dY = y-referenceY;

			int distance = dX*dX+dY*dY;
				if(distance>=bandMinRadiusSquared&&distance<bandMaxRadiusSquared)
				{
					meanIn+=image[x+y*horizontalSize*32];
					meanCountIn++;
				}
				else
				{
					meanOut+=image[x+y*horizontalSize*32];
					meanCountOut++;
				}
			
		}
	}

	int outOffset = horizontalSize*verticalSize;
	means[index] = meanIn;
	means[index+outOffset]=meanOut;
	meanCounts[index] = meanCountIn;
	meanCounts[index+outOffset]=meanCountOut;
}

// function for creating the selected amount of Gabor filters matrices
__global__ void makeFilters(float frequency, float* result,int amount)
{
	int index = threadIdx.x +
		threadIdx.y*32+blockIdx.x*32*32;

	float deltaX = 1.0f/frequency/2.3f;
	float deltaY = 1.0f/frequency/2.3f;

	float xCenter = 16.0f;
	float yCenter = 16.0f;
	float angle = CUDART_PI_F/amount*blockIdx.x;
	float sinAngle=sin(angle), cosAngle = cos(angle);
	
	int dX = threadIdx.x - xCenter,dY = threadIdx.y - yCenter;

	float xDash = sinAngle*dX + cosAngle*dY, yDash = cosAngle*dX - sinAngle*dY;

	float cellExp=exp(-0.5f*(xDash*xDash/deltaX/deltaX+ yDash*yDash/deltaY/deltaY));
	float cellCos=cos(2.0f*CUDART_PI_F*frequency*xDash);

	result[index] = cellExp*cellCos;
}

__global__ void formRawOrientationFieldComponents(
	int* sobelXField,
	int* sobelYField,
	float* orFieldXComponent,
	float* orFieldYComponent,
	int pitch,
	int width,
	int height,
	int regionSize)
{
	int orFieldXDim = width/(regionSize - 1);
    int orFieldYDim = height/(regionSize - 1);
	int regionX = blockIdx.x*32+threadIdx.x;
	int regionY = blockIdx.y*32+threadIdx.y;

	if(regionX<orFieldXDim&&regionY<orFieldYDim)
	{
		float xx = 0.0f,
		 	  yy=0.0f;
		for (int u = 0; u < regionSize; u++)
        {
            for (int v = 0; v < regionSize; v++)
            {
                 int mX = regionX*(regionSize - 1) + u;
                 int mY = regionY*(regionSize - 1) + v;
                 if (mX > width || mY > height) continue;
				 int sobelIndex = mX+mY*width;
				 float sx = sobelXField[sobelIndex];
				 float sy = sobelYField[sobelIndex];
                 yy += 2.0f*sx*sy;
                 xx += -sx*sx +sy*sy;
            }
        }
        float hypotenuse = sqrt(xx*xx + yy*yy);
		int regionIndex = regionX+pitch/sizeof(float)*regionY;
        orFieldXComponent[regionIndex] = yy/hypotenuse;
        orFieldYComponent[regionIndex] = xx/hypotenuse;
	}
}

__global__ void filterImage(
	int width,
	int height,
	int* imgBytes,
	int* resultBytes,
	int referenceX,
	int referenceY,
	int holeRadius,
	int bandRadius,
	int bandNumber,
	float* filterPointer)
{
	int x = blockIdx.x*32+threadIdx.x;
	int y = blockIdx.y*32+threadIdx.y;
	int index = blockIdx.z*width*height+ x+width*y;
	int diffX = x-referenceX;
	int diffY = y-referenceY;

	int distance = diffX*diffX + diffY*diffY;

	int max = holeRadius + bandNumber*bandRadius;
	max*=max;

	if(distance<=max&&distance>=holeRadius*holeRadius)
	{
		float color =0.0f;
		for(int i=-16;i<16;i++)
		{
			for(int j=-16;j<16;j++)
			{
				float multiple = filterPointer[blockIdx.z*32*32+32*(16+j)+16+i];
				//if(abs(multiple)>=0.05)
					color+=multiple*(float)imgBytes[(y+j)*width+i+x];
			}
		}
		if(color<0)color=0.0f;
		if(color>255)color=255.0f;
		resultBytes[index]=(int)color;
	}
		
	else resultBytes[index]=0;
}

// init function should be called first
int initDevice()
{
	cudaError_t cudaStatus;
	database =0;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        return 1;
    }
	return 0;
}

// dispose function, should be called last
int disposeDevice()
{
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	if(cudaFilterPointer!=0)
	{
		cudaFree(&cudaFilterPointer);
	}

	if(database!=0)
	{
		cudaFree(&database);
	}

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        return 1;
    }

    return 0;
}

void internalNormalizeImage(
	int* image, // the image itself - already stored in cuda
	int width, // its width
	int height, // height
	int referenceX, //reference point location
	int referenceY, 
	int bandMinRadius, // inner band radius
	int bandMaxRadius, // outer band radius
	int baseMean, // the base mean of normalized areas
	int baseVariance) //the base variance
{
	float* means;
	float* variances;
	int* meanCounts;

	int verticalSize = height/32;
	int horizontalSize = width/32;

	int size = horizontalSize*verticalSize; //32*32
	
	float* variancesCpu = (float*)malloc(2*size*sizeof(float));
	float* meansCpu = (float*)malloc(2*size*sizeof(float));
	int* meanCountsCpu = (int*)malloc(2*size*sizeof(int));

	cudaError_t error = cudaMalloc(&means, 2*size*sizeof(float));
	error = cudaMalloc(&variances, 2*size*sizeof(float));
	error =cudaMalloc(&meanCounts, 2*size*sizeof(int));

	countMeans<<<1,dim3(horizontalSize,verticalSize)>>>(
		image,
		means,
		meanCounts,
		32,
		referenceX,
		referenceY,
		bandMinRadius*bandMinRadius,
		bandMaxRadius*bandMaxRadius,
		horizontalSize, 
		verticalSize);

	error = cudaGetLastError();

	float meanIn = 0;
	float meanInCount = 0;
	float meanOut = 0;
	float meanOutCount = 0;
	error = cudaMemcpy(meansCpu,means, 2*size*sizeof(float),cudaMemcpyDeviceToHost);
	error = cudaMemcpy(meanCountsCpu,meanCounts, 2*size*sizeof(int),cudaMemcpyDeviceToHost);
	int offset = verticalSize*horizontalSize;
	for(int i=0;i<offset;i++)
	{
		meanIn+=meansCpu[i];
		meanOut+=meansCpu[offset+i];
		meanInCount += meanCountsCpu[i];
		meanOutCount += meanCountsCpu[i+offset];
	}

	meanIn/=meanInCount;
	meanOut/=meanOutCount;

	countVariances<<<1,dim3(horizontalSize,verticalSize)>>>(
		image,
		variances,
		meanIn,
		meanOut,
		32,
		referenceX,
		referenceY,
		bandMinRadius*bandMinRadius,
		bandMaxRadius*bandMaxRadius,
		horizontalSize,
		verticalSize);

	error = cudaGetLastError();

	error = cudaMemcpy(variancesCpu,variances, 2*size*sizeof(float),cudaMemcpyDeviceToHost);
	float varianceIn = 0;
	float varianceOut = 0;
	for(int i=0;i<offset;i++)
	{
		varianceIn+=variancesCpu[i];
		varianceOut+=variancesCpu[offset+i];
	}
	varianceIn/=meanInCount;
	varianceOut/=meanOutCount;

	applyNormalization<<<dim3(horizontalSize,verticalSize),dim3(32,32)>>>(
		image,
		baseMean,
		baseVariance,
		meanIn,
		varianceIn,
		meanOut,
		varianceOut,
		referenceX,
		referenceY,
		bandMinRadius*bandMinRadius,
		bandMaxRadius*bandMaxRadius,
		horizontalSize
		);
	error = cudaGetLastError();

	/*FILE* file = fopen("C:\\temp\\nimage.bin","rb");
	int* imageBack = (int*)malloc(sizeof(int)*width*height);
	int* imageFile = (int*)malloc(sizeof(int)*width*height);
	int read = fread(imageFile,sizeof(int),width*height,file);
	fclose(file);

	error = cudaMemcpy2D(imageBack,width*sizeof(int),image,width*sizeof(int),width*sizeof(int),height,cudaMemcpyDeviceToHost);

	for(int i=0;i<width*height;i++)
	{
		if(abs(imageBack[i]-imageFile[i])>0)
		{
			i=i;
		}
	}

	free(imageBack);
	free(imageFile);*/
}

void internalCreateGaborFilters(float frequency, int amount)
{
	int size = amount*32*32;
	//fprintf(stdout,"Size is %d\n",size);
	cudaError_t error = cudaMalloc((void**)&cudaFilterPointer,size*sizeof(float));
	if(error!=cudaSuccess)
	{
		//fprintf(stdout,"Malloc error\n");
		cudaFree(&cudaFilterPointer);
		return;
	}
  
	makeFilters<<<dim3(amount), dim3(32,32)>>>(frequency,cudaFilterPointer,amount);

	error = cudaGetLastError();
	//if(error==cudaSuccess)
	//	fprintf(stdout, "Filters created\n");
}

// exported function that creates the orientation field and searches for the core point
void findCorePoint(int* imgBytes, int width, int height, int* xCore, int* yCore)
{
	void* cudaImagePointer = 0;
	void* cudaSobelPointerX = 0;
	void* cudaSobelPointerY = 0;


	size_t pitch1=0, pitch2=0;
	cudaError_t error;
	error = cudaMallocPitch((void**)&cudaImagePointer,&pitch1,width*sizeof(int),height);

	error = cudaMallocPitch((void**)&cudaSobelPointerX,&pitch2,width*sizeof(int),height);

	error = cudaMallocPitch((void**)&cudaSobelPointerY,&pitch2,width*sizeof(int),height);

	error = cudaMemcpy2D(cudaImagePointer,pitch1,imgBytes,width*sizeof(int),width*sizeof(int),height,cudaMemcpyHostToDevice);

	applySobel<<<dim3(width/32,height/32),dim3(32,32)>>>((int*)cudaImagePointer,(int*)cudaSobelPointerX,
		(int*)cudaSobelPointerY,width, height);

	error = cudaGetLastError();

    // sobel is working
	int regionSize = 10;
    int orFieldXDim = width/(regionSize - 1);
    int orFieldYDim = height/(regionSize - 1);
	void* cudaOrFieldPointer = 0;
	void* cudaOrFieldXPointer = 0;
	void* cudaOrFieldYPointer = 0;
	void* cudaCoreMetricPointer = 0;
	size_t pitch3=0, pitch4=0;
	size_t pitch5=0, pitch6=0;
	error = cudaMallocPitch((void**)&cudaOrFieldPointer,&pitch3,orFieldXDim*sizeof(float),orFieldYDim);
	error = cudaMallocPitch((void**)&cudaOrFieldXPointer,&pitch4,orFieldXDim*sizeof(float),orFieldYDim);
	error = cudaMallocPitch((void**)&cudaOrFieldYPointer,&pitch5,orFieldXDim*sizeof(float),orFieldYDim);
	error = cudaMallocPitch((void**)&cudaCoreMetricPointer,&pitch6,orFieldXDim*sizeof(float),orFieldYDim);

	formRawOrientationFieldComponents<<<dim3(orFieldXDim/32+1,orFieldYDim/32+1),dim3(32,32)>>>(
		(int*)cudaSobelPointerX, (int*)cudaSobelPointerY, (float*)cudaOrFieldXPointer,(float*)cudaOrFieldYPointer,
		pitch3,width,height,regionSize);

	error = cudaGetLastError();
	/*FILE* file = fopen("D:\\prevx.bin","rb");
	float* of = (float*)malloc(sizeof(float)*orFieldXDim*orFieldYDim);
	int read = fread(of,sizeof(float),orFieldXDim*orFieldYDim,file);
	fclose(file);
	float* ofCuda = (float*)malloc(sizeof(float)*orFieldXDim*orFieldYDim);

	error = cudaMemcpy2D(ofCuda,orFieldXDim*sizeof(float),cudaOrFieldXPointer,pitch4,orFieldXDim*sizeof(float),orFieldYDim,cudaMemcpyDeviceToHost);

	for(int i=0;i<orFieldXDim*orFieldYDim;i++)
	{
		if(__isnanf(of[i])^__isnanf(ofCuda[i])|| abs(of[i]-ofCuda[i])>0.00001)
		{
			i=i;
		}
	}

	free(of);
	free(ofCuda);*/
	// looks like raw is formed ok as well - but probably need more investigation

	makeSmoothedOrientationField<<<dim3(orFieldXDim/32+1,orFieldYDim/32+1),dim3(32,32)>>>(
		(float*) cudaOrFieldPointer, (float*)cudaOrFieldXPointer,(float*)cudaOrFieldYPointer, pitch3,
		orFieldXDim,orFieldYDim);

	error = cudaGetLastError();

	/*FILE* file = fopen("D:\\orfield.bin","rb");
	float* of = (float*)malloc(sizeof(float)*orFieldXDim*orFieldYDim);
	int read = fread(of,sizeof(float),orFieldXDim*orFieldYDim,file);
	fclose(file);
	float* ofCuda = (float*)malloc(sizeof(float)*orFieldXDim*orFieldYDim);

	error = cudaMemcpy2D(ofCuda,orFieldXDim*sizeof(float),cudaOrFieldPointer,pitch3,orFieldXDim*sizeof(float),orFieldYDim,cudaMemcpyDeviceToHost);

	for(int i=0;i<orFieldXDim*orFieldYDim;i++)
	{
		if(__isnanf(of[i])^__isnanf(ofCuda[i])|| abs(of[i]-ofCuda[i])>0.00001)
		{
			i=i;
		}
	}

	free(of);
	free(ofCuda);*/

	// OK GUSY ZEE ORFIELD GENERATOR WORKS FINE - I guess...

	calculateCoreMetric<<<dim3(orFieldXDim/32+1,orFieldYDim/32+1),dim3(32,32)>>>(
		(float*) cudaOrFieldPointer, (float*) cudaCoreMetricPointer, pitch3/sizeof(float), orFieldXDim,orFieldYDim);

	error = cudaGetLastError();

	float* metric = (float*)malloc(sizeof(float)*orFieldXDim*orFieldYDim);
	error = cudaMemcpy2D(metric,orFieldXDim*sizeof(float),cudaCoreMetricPointer,pitch3,orFieldXDim*sizeof(float),orFieldYDim,cudaMemcpyDeviceToHost);
	int maxIndex = 0;
	int maxMetric = -100500.0f;
	for(int i=0;i<orFieldXDim*orFieldYDim;i++)
	{
		if(metric[i]>maxMetric)
		{
			maxMetric = metric[i];
			maxIndex = i;
		}
	}
	*xCore = (int)(((float)(maxIndex%orFieldXDim)+0.5f)*(regionSize-1));
	*yCore = (int)(((float)(maxIndex/orFieldXDim)+0.5f)*(regionSize-1));;

	error = cudaFree(cudaImagePointer);

	error = cudaFree(cudaOrFieldPointer);

	error = cudaFree(cudaOrFieldXPointer);

	error = cudaFree(cudaOrFieldYPointer);

	error = cudaFree(cudaCoreMetricPointer);

	error = cudaFree(cudaSobelPointerX);

	error = cudaFree(cudaSobelPointerY);
}

// function for building an actual FCode from P filtered images
void processFilteredImage(int* imgBytes, float* result, int width, int height, int filterAmount, int numBands, int numSectors,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY)
{
            double* means =  (double*)malloc(sizeof(double)*numBands*numSectors*filterAmount);
			double* variances =  (double*)malloc(sizeof(double)*numBands*numSectors*filterAmount);
			int* meanCounts =  (int*)malloc(sizeof(int)*numBands*numSectors*filterAmount);


			for(int i=0;i<numBands*numSectors*filterAmount;i++)variances[i]=means[i]=result[i]=meanCounts[i]=0;//memset???
            int* bucketHash = (int*)malloc(width*height*sizeof(int));
            
			int filterRadius = holeRadius + numBands*bandRadius;
            // divide by sectors
            for (int x = referencePointX-filterRadius; x <= referencePointX+filterRadius; x++)
            {
                for (int y = referencePointY-filterRadius; y <= referencePointY+filterRadius; y++)
                {
					int X = x - referencePointX;
                    int Y = y - referencePointY;
                    double radius = sqrt((double)X*X + Y*Y);
                    if (radius < holeRadius || radius >= filterRadius)
                    {
                        bucketHash[x+ y*width] =-1;
						continue;
                    }
                    int bandBase = (int) ((radius - holeRadius)/bandRadius);

					double angle = (X == 0) ? (Y > 0) ? (CUDART_PI/2) : (-CUDART_PI/2) : atan((double)Y/X);
                    if (X < 0) angle += CUDART_PI;
                    if (angle < 0) angle += 2.0*CUDART_PI;
                    int sectorNumber = (int) (angle/(CUDART_PI*2.0/numSectors));
                    int currentSector = bucketHash[x+ y*width] = bandBase*numSectors + sectorNumber;
					for(int filter=0;filter<filterAmount;filter++)
					{
						means[currentSector+filter*numBands*numSectors]+=
							imgBytes[filter*width*height+y*width+x];
						meanCounts[currentSector+filter*numBands*numSectors]++;
					}
                }
            }

			for(int i=0;i<numBands*numSectors*filterAmount;i++)means[i]/=meanCounts[i];

			for (int x = referencePointX-filterRadius; x <= referencePointX+filterRadius; x++)
            {
                for (int y = referencePointY-filterRadius; y <= referencePointY+filterRadius; y++)
                {
					int index = x+width*y;
					int currentSector = bucketHash[index];
					if(currentSector!=-1)
					{
						for(int filter=0;filter<filterAmount;filter++)
						{
							variances[currentSector+filter*numBands*numSectors]+=
								abs((double)imgBytes[index+filter*width*height]-means[currentSector+filter*numBands*numSectors]);
						}
					}
				}
			}
            
			for(int i=0;i<numBands*numSectors*filterAmount;i++)result[i]=(float)(variances[i]/meanCounts[i]);

			free(bucketHash);
            free(meanCounts);
			free(means);
}

void normalizeImage(int* imgBytes, int* result, int width, int height, int numBands,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY)
{
	size_t pitch1=0;
	void* cudaImagePointer = 0;
	cudaError_t error;
	error = cudaMallocPitch((void**)&cudaImagePointer,&pitch1,width*sizeof(int),height);
	
	error = cudaMemcpy2D(cudaImagePointer,pitch1,imgBytes,width*sizeof(int),width*sizeof(int),height,cudaMemcpyHostToDevice);

	internalNormalizeImage(
		(int*)cudaImagePointer,
		width,
		height,
		referencePointX,
		referencePointY,
		holeRadius,
		holeRadius+numBands*bandRadius,
		100.0f,
		100.0f);

	error = cudaMemcpy2D(result,width*sizeof(int),cudaImagePointer,pitch1,width*sizeof(int),height,cudaMemcpyDeviceToHost);

	cudaFree(cudaImagePointer);
}

void fullFilterImage(int* imgBytes, int* result, int width, int height, 
													int filterAmount, int numBands,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY)
{
	if(cudaFilterPointer==0)
	{
		internalCreateGaborFilters(0.1,filterAmount);
	}

	size_t pitch1=0, pitch2=0;
	void* cudaImagePointer = 0;
	void* cudaImageResultPointer = 0;
	cudaError_t error;
	error = cudaMallocPitch((void**)&cudaImagePointer,&pitch1,width*sizeof(int),height*filterAmount);

	error = cudaMallocPitch((void**)&cudaImageResultPointer,&pitch2,width*sizeof(int),height*filterAmount);
	
	error = cudaMemcpy2D(cudaImagePointer,pitch1,imgBytes,width*sizeof(int),width*sizeof(int),height,cudaMemcpyHostToDevice);


	internalNormalizeImage(
		(int*)cudaImagePointer,
		width,
		height,
		referencePointX,
		referencePointY,
		holeRadius,
		holeRadius+numBands*bandRadius,
		100.0f,
		100.0f);

	filterImage<<<dim3(640/32,480/32,filterAmount),dim3(32,32)>>>(
		width,
		height,
		(int*)cudaImagePointer,
		(int*)cudaImageResultPointer,
		referencePointX,
		referencePointY,
		holeRadius,
		bandRadius,
		numBands,
		cudaFilterPointer);

	error = cudaGetLastError();

	error = cudaMemcpy2D(result,width*sizeof(int),cudaImageResultPointer,pitch2,width*sizeof(int),height*filterAmount,cudaMemcpyDeviceToHost);

	error = cudaFree(cudaImagePointer);

	error = cudaFree(cudaImageResultPointer);
}

// exported function for the full cycle of FingerCode creation: normalizing, filtering and making the actual vector
void createFingerCode(int* imgBytes, float* result, int width, int height, int filterAmount, int numBands, int numSectors,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY)
{
	
	int* fullImage = (int*)malloc(height*width*filterAmount*sizeof(int));

	fullFilterImage(imgBytes,fullImage,width,height,filterAmount,numBands,holeRadius,bandRadius,referencePointX,referencePointY);

	

	processFilteredImage(fullImage, result, width, height, filterAmount, numBands, numSectors,
                                                    holeRadius, bandRadius,  referencePointX,
                                                    referencePointY);

	free(fullImage);
}

// load the fingerprint database, where the codes are stored normally, i.e. FCode1, FCode2...
void loadNoncoalescedFingerCodeDatabase(float* fingers, int numberOfRegions, int numberOfCodes)
{
	int length = (numberOfCodes/16)*16;
	if(numberOfCodes%16!=0)length+=16;
	float* coalesced = (float*) malloc(sizeof(float)*numberOfRegions*length);
	int offset = 16*numberOfRegions;
	for(int i=0;i<numberOfCodes;i++)
	{
		int warpOffset = i%16;
		int warp = i/16;
		for(int j=0;j<numberOfRegions;j++)
		{
			coalesced[warp*numberOfRegions*16+16*j+warpOffset] = fingers[i*numberOfRegions+j];
		}
	}
	loadFingerCodeDatabase(coalesced,numberOfRegions,numberOfCodes);
	free(coalesced);
}

// load the fingerprint database, where the codes are stored in coalescing optimized mode: 
// FCode1[0], FCode2[0],..., FCode1[n], FCode2[n]...
void loadFingerCodeDatabase(float* fingers, int numberOfRegions, int numberOfCodes)
{
	cudaError_t error;
	databaseCount = numberOfCodes;
	error = cudaMalloc((void**)&database, sizeof(float)*numberOfRegions*numberOfCodes);

	error = cudaMemcpy(database,fingers,sizeof(float)*numberOfRegions*numberOfCodes,cudaMemcpyHostToDevice);

}

//assuming result array is already allocated
void matchFingercode(float* fingercode, float numberOfRegions, float* result)
{
	cudaError_t error;
	float* cudaFingercode;
	float* cudaResult;
	error = cudaMalloc((void**)&cudaFingercode, sizeof(float)*numberOfRegions);

	error = cudaMalloc((void**)&cudaResult, sizeof(float)*databaseCount);

	error = cudaMemcpy(cudaFingercode,fingercode,sizeof(float)*numberOfRegions,cudaMemcpyHostToDevice);

	match<<<dim3(databaseCount/1024+1),dim3(32,32)>>>(cudaFingercode, database,numberOfRegions,databaseCount, cudaResult);

	error = cudaGetLastError();

	error = cudaFree(cudaFingercode);


	error = cudaMemcpy(result,cudaResult,sizeof(float)*databaseCount,cudaMemcpyDeviceToHost);

	error = cudaFree(cudaResult);
}

int createGaborFilters(float* result, float frequency, int amount)
{
	internalCreateGaborFilters(frequency, amount);
	int size = amount*32*32;
	cudaError_t error = cudaMemcpy(result,cudaFilterPointer,size*sizeof(float),cudaMemcpyDeviceToHost);
	if(error!=cudaSuccess)
	{
		cudaFree(&cudaFilterPointer);
		cudaFilterPointer = 0;
		return -1;
	}

	return size;
}


// main function is for debugging the library as a standalone application
int main()
{
	initDevice();

	int K = 1000000;
	float* arr= (float*)malloc(K*sizeof(float));
	int* indexes= (int*)malloc(K*sizeof(int));
	srand(K);
	for(int i=0;i<K;i++)
	{
		arr[i]=(float)rand()/100.f;
		indexes[i]=i;
	}

	sortArrayAndIndexes(arr,indexes,K);
	int o=0;
	/*int size = 80;

	int num = 60000;

	float* dbase = (float*)malloc(sizeof(float)*num*size);

	for(int i=0;i<num*size;i++)
	{
		dbase[i]=0.0;
	}

	float* sample = (float*)malloc(sizeof(float)*size);

	for(int i=0;i<size;i++)
	{
		sample[i]=0.0;
	}

	float* result = (float*)malloc(sizeof(float)*num);

	loadFingerCodeDatabase(dbase,size,num);

	matchFingercode(sample,size,result);*/

	//FILE* file = fopen("C:\\temp\\array.bin","rb");
	//int* imgBytes = (int*)malloc(640*480*sizeof(int));

	//int bytesRead = fread(imgBytes,sizeof(int),640*480,file);

	//int x=0;
	//int y=0;
	//fclose(file);

	////findCorePoint(imgBytes,640,480,&x,&y);
	//int numBands = 3;
	//int numSectors = 16;
	//int filterAmount = 8;
	//int holeRadius = 20;
	//int bandRadius = 20;

	//float* result = (float*)malloc(sizeof(float)*numBands*filterAmount*numSectors);

	//createFingerCode(imgBytes,result,640,480,filterAmount,numBands,numSectors,holeRadius,bandRadius,320,240);

	/*file = fopen("C:\\temp\\fcode.bin","rb");
	double* fcodeFile = (double*)malloc(8*3*16*sizeof(double));

	bytesRead = fread(fcodeFile,sizeof(double),8*3*16,file);

	fclose(file);

	for(int i=0;i<8*3*16;i++)
	{
		if(abs(fcodeFile[i]-result[i])>0.001)
		{
			i=i;
		}
	}*/

	disposeDevice();
}