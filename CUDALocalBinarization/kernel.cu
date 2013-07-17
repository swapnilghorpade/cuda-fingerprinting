
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CUDAArray.cuh"

cudaError_t localBinarizationCuda(float *img_dst, float *img_src, float* img_win,int width, int height);

__device__ int sgn(float num)
{
	if(num > 0){
		return 1;
	} else if(num < 0) {
		return -1;
	} else {
		return 0;
	}
}

__global__ void sobel(float* img_dst, float* img_src, int width, int height, float* gX, float* gY, float* theta, size_t devPitch, size_t pitchMask) 
{
	float PI = 3.14159265359;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	int index = i + j*devPitch;

	if((i > -1) && (i < width) && (j > -1) && (j < height))
	{
		theta[index] = 0;
	}

	if((i > 0) && (i < width - 1) && (j > 0) && (j < height - 1))
	{
		float newX = 0;
		float newY = 0;

		for (int h = 0; h < 3; h++)
        {
            for (int w = 0; w < 3; w++)
            {
				float curr = img_src[i + h - 1 + (j + w - 1)*devPitch];
                newX += gX[h + w*pitchMask] * curr;
                newY += gY[h + w*pitchMask] * curr;
            }
        }

		img_dst[index] = sqrt(newX * newX + newY * newY);

		if (newX == 0)
        {
            theta[index] = 90;
        }
        else
        {
			theta[index] = atan(newY / newX)*(180 / PI);
        }
	} else if(i == 0) {
		if(j > -1 && j < height) {
			img_dst[index] = 255;
			theta[i + j*devPitch] = 0;
		}
	} else if(i == width - 1) {
		if(j > -1 && j < height) {
			img_dst[index] = 255;
			theta[i + j*devPitch] = 0;
		}
	} else if(j == 0) {
		if(i > -1 && i < width) {
			img_dst[index] = 255;
			theta[i + j*devPitch] = 0;
		}
	} else if(j == height - 1) {
		if(i > -1 && i < width) {
			img_dst[index] = 255;
			theta[i + j*devPitch] = 0;
		}
	}
}

__global__ void nonMaximumSuppression(float* img_dst, float* img_src, float* theta, int width, int height, size_t devPitch) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	int index = i + j*devPitch;

	if((i > -1) && (i < width) && (j > -1) && (j < height))
	{
		img_dst[i + j*devPitch] = 0;
	}

	if((i > -1) && (i < width) && (j > -1) && (j < height))
	{
		//img_dst[i + j*devPitch] = 0;
		if (theta[index] > 67.5)
        {
            if (theta[index] > 112.5)
            {
                if (theta[index] > 157.5)
                {
                    theta[index] = 135;
                }
                else
                {
                    theta[index] = 0;
                }
            }
            else
            {
                theta[index] = 90;
            }
        }
        else
        {
            if (theta[index] > 22.5)
            {
                theta[index] = 45;
            }
            else
            {
				theta[index] = 0;
            }
        }

		float dxCos = cos(theta[index]);
		float dySin = sin(theta[index]);

		int dx = sgn(dxCos);
		int dy = -sgn(dySin);

		if(i == 0) {
			if(j > -1 && j < height) {
				img_dst[index] = 0;
			}
		} else if(i == width - 1) {
			if(j > -1 && j < height) {
				img_dst[index] = 0;
			}
		} else if(j == 0) {
			if(i > -1 && i < width) {
				img_dst[index] = 0;
			}
		} else if(j == height - 1) {
			if(i > -1 && i < width) {
				img_dst[index] = 0;
			}
		} else {//i == 173 && j == 63
			if (img_src[index] > img_src[(i + dx) + (j + dy)*devPitch] 
			&& img_src[index] > img_src[(i - dx) + (j - dy)*devPitch])
			{
				img_dst[index] = img_src[index];
			} else {
				img_dst[index] = 0;//
			}//иначе остаётся 0
		}
	} 
}

__global__ void globalBinarisationInv(float border, float *img_dst, float *img_src, int width, int height, size_t pitch)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i > -1 && i < width && j > -1 && j < height)
	{
		//img_dst[i + j*pitch] = img_src[i + j*pitch] > border ? 255 : 0;
		img_dst[i + j*pitch] = 0;
		if(img_src[i + j*pitch] > border) 
		{
			img_dst[i + j*pitch] = 0;
		} else {
			img_dst[i + j*pitch] = 255;
		}
	}
}

/*__global__ void localBinarization(float* img_dst, float* imgWin, float* imgBorderWin, int widthImg, int heightImg, int widthWinImg, int heightWinImg,
	int winWidth, int winHeight, int winWidScale, int winHeiScale, int numberSubImgX, int numberSubImgY, size_t pitchDev, size_t pitchWinScale, size_t pitchColor)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;



}*/

__global__ void initializationValue(float* dst, float value, int width, int height, size_t pitch)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < width && j < height) {
		dst[i + j*pitch] = value;
	}
}

__global__ void findMostDarkestColor(float* darkestColor, float* img_src, float* imgBorder, int width, int height, int winWidth, int winHeight, size_t pitch, size_t pitchColor) 
{
	int column = blockIdx.x;//*blockDim.x + threadIdx.x;
	int row = blockIdx.y;//*blockDim.y + threadIdx.y;
	
//	int column = threadIdx.x;
//	int row = threadIdx.y;
	//теперь индексация не по всей картинке!!!
	int indexColor = column + row*pitchColor;
	for (int i = 0; i < winHeight; i++)
    {
        for (int j = 0; j < winWidth; j++)
		{
			int indexImg = column*winWidth+i + (j + row*winHeight)*pitch;
            if (imgBorder[indexImg] < 1)
            {
				if(darkestColor[indexColor] < 0) {
					darkestColor[indexColor] = img_src[indexImg];
				} else {
					if(darkestColor[indexColor] > img_src[indexImg]) {
						darkestColor[indexColor] = img_src[indexImg];
					}
				}    
			}
        }
    }

	/*if(column < winWidth && row < winHeight) {
		if (imgBorder[i + j*pitch] < 1)
		{
			__threadfence_block();
			if(darkestColor[blockIdx.x + blockIdx.y*pitchColor] < 0) {
				darkestColor[blockIdx.x + blockIdx.y*pitchColor] = img_src[i + j*pitch];
			} else {
				if(darkestColor[blockIdx.x + blockIdx.y*pitchColor] > img_src[i + j*pitch]) {
					darkestColor[blockIdx.x + blockIdx.y*pitchColor] = img_src[i + j*pitch];
				}
			}
		}
	}*/
}

__global__ void clarificationImg(float* img_dst, float* img_src, float border, int width, int height, int winWidth, int winHeight, size_t pitch) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	int column = threadIdx.x;
	int row = threadIdx.y;
	if(i < width && j < height) {
		if(column < winWidth && row < winHeight) {
			if(img_src[i + j*pitch] > border) {
				img_dst[i + j*pitch] = 255;
			} else {
				img_dst[i + j*pitch] = 0;
			}
		}
	}
}

__global__ void fillingImg(float* img_dst, float* img_src, float* darkestImg, int width, int height, int winWidth, int winHeight, size_t pitch, size_t pitchColor) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	int column = threadIdx.x;
	int row = threadIdx.y;
	if(i < width && j < height) {
		if(column < winWidth && row < winHeight) {
			if(img_src[i + j*pitch] < darkestImg[blockIdx.x + blockIdx.y*pitchColor]) {
				img_dst[i + j*pitch] = 0;
			} else {
				img_dst[i + j*pitch] = 255;
			}
		}
	}
}

__global__ void combinResBinImgs(float* img_dst, float* img_first, float* img_second, int widthSRC, int heightSRC, int winWidthRes, int winHeightRes, int winDiffW, int winDiffH, size_t pitch, size_t pitchWin) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < widthSRC && j < heightSRC) {
		int column = threadIdx.x;
		int row = threadIdx.y;

		if(column < winWidthRes && row < winHeightRes) {
			int iRes = blockIdx.x*(blockDim.x - winDiffW) + threadIdx.x;
			int jRes = blockIdx.y*(blockDim.y - winDiffH) + threadIdx.y;

			if(img_first[i + j*pitchWin] < 1 || img_second[i + j*pitchWin] < 1 ) {
				img_dst[iRes + jRes*pitch] = 0;
			} else {
				img_dst[iRes + jRes*pitch] = 255;
			}

		}

	}
}

__global__ void copyImgToImgWin(float* imgWin, float* img, int width, int height, int winWidth, int winHeight, int diffWidth, int diffHeight, size_t pitch, size_t pitchWin) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	int iSW = blockIdx.x*(blockDim.x + diffWidth) + threadIdx.x;
	int jSW = blockIdx.y*(blockDim.y + diffHeight) + threadIdx.y;
	
	//imgWin[iSW + jSW*pitchWin] = 0;
	
	if(iSW == 0 && jSW == 0) {
		imgWin[iSW + jSW*pitchWin] = 45;
	}

/*	if(threadIdx.x < diffWidth) {
		imgWin[iSW + winWidth + jSW*pitchWin] = 0;
	}
	
	if(threadIdx.y < diffHeight) {
		imgWin[iSW + (jSW + winHeight)*pitchWin] = 0;
	}
	//return;
	if(threadIdx.x < diffWidth && threadIdx.y < diffHeight) {
		imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = 0;
	}

	return;*/

	if(threadIdx.x < winWidth && threadIdx.y < winHeight) {
		if(i < width && j < height) {
			imgWin[iSW + jSW*pitchWin] = img[i + j*pitch];
	/*		if(threadIdx.y < diffHeight) {
				imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = 0;
			}
			if(threadIdx.x < diffWidth) {
				imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = 0;
			}*/

			if(threadIdx.x < diffWidth) {
				int ii = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
				if(ii >= width) {
					imgWin[iSW + winWidth + jSW*pitchWin] = 0;
				} else {
					imgWin[iSW + winWidth + jSW*pitchWin] = img[ii + j*pitch];
					if(threadIdx.y < diffHeight) {
						int jj = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
						if(jj >= height) {
							imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = 0;
						} else {
							imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = img[ii + jj*pitch];
						}
					}
				}
			}

			if(threadIdx.y < diffHeight) {
				int jj = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
				if(jj >= height) {
					imgWin[iSW + (winHeight + jSW)*pitchWin] = 0;
				} else {
					imgWin[iSW + (winHeight + jSW)*pitchWin] = img[i + jj*pitch];
				}
			}

		}
	
	/*	if(threadIdx.x < diffWidth) {
			int ii = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
			if(ii > width) {
				imgWin[iSW + winWidth + jSW*pitchWin] = 0;
				if(threadIdx.y < diffHeight) {
					imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = 0;
				}
			} else {
				imgWin[iSW + winWidth + jSW*pitchWin] = img[ii + j*pitch];
				if(threadIdx.y < diffHeight) {
					int jj = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
					if(jj >= height) {
						imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = 0;
					} else {
						imgWin[iSW + winWidth + (jSW + winHeight)*pitchWin] = img[ii + jj*pitch];
					}
				}
			}
		}

		if(threadIdx.y < diffHeight) {
			int jj = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
			if(jj > height) {
				imgWin[iSW + (winHeight + jSW)*pitchWin] = 0;
			} else {
				imgWin[iSW + (winHeight + jSW)*pitchWin] = img[i + jj*pitch];
			}
		}*/
	}

}

__global__ void emptyImg(float* img_dst, int width, int height, size_t pitch) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < width && j < height) {
		img_dst[i + j*pitch] = 0;
	}
	if(i == 0 && j == 0) {
		img_dst[i + j*pitch] = 255;
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
	int width = 20;
	int height = 20;

	CUDAArray<float> img = loadImage("C:\\temp\\104_6.bin", true);
	width = img.Width;
	height = img.Height;

	float* img_src;// = (float*) malloc(sizeof(float)*width*height);
	float* img_dst = (float*) malloc(sizeof(float)*width*height);

	img_src = img.cpuP;

	int numberSubImgX = width/16 + (width%16 > 0 ? 1 : 0);
    int numberSubImgY = height/16 + (height%16 > 0 ? 1 : 0);

	int winWidScale = 16*1.3;
	int winHeiScale = 16*1.3;

	float* img_win = (float*) malloc(sizeof(float)*winWidScale*winHeiScale*numberSubImgX*numberSubImgY);

    // Add vectors in parallel.
	cudaError_t cudaStatus = localBinarizationCuda(img_dst, img_src, img_win, width, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	SaveArray(img_dst, width, height,  "C:\\temp\\104_6_2.bin");

	//SaveArray(img_win, winWidScale*numberSubImgX, winHeiScale*numberSubImgY,  "C:\\temp\\104_6_2.bin");

	//free(img_dst);
	free(img_src);
	free(img_win); /// мб его удалить?
	
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t localBinarizationCuda(float *img_dst, float *img_src, float* img_win,  int width, int height)
{
    float *dev_src = 0;
    float *dev_dst = 0;
	float *dev_imgBorder = 0;
	float *dev_theta = 0;
	float *dev_sobel = 0;

	size_t pitch;
	size_t prevPitch;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
	cudaStatus = cudaMallocPitch((void **)&dev_dst, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	prevPitch = pitch;

	cudaStatus = cudaMallocPitch((void**)&dev_src, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	if(prevPitch != pitch) {
		printf("inoi pitch!!!\n");
	}

	cudaStatus = cudaMallocPitch((void**)&dev_imgBorder, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	if(prevPitch != pitch) {
		printf("inoi pitch!!!\n");
	}

	cudaStatus = cudaMallocPitch((void**)&dev_theta, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	if(prevPitch != pitch) {
		printf("inoi pitch!!!\n");
	}

	cudaStatus = cudaMallocPitch((void**)&dev_sobel, &pitch, width*sizeof(float), height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	if(prevPitch != pitch) {
		printf("inoi pitch!!!\n");
	}

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy2D(dev_src, pitch, img_src, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	float sobelMaskX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1};
	float sobelMaskY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1};

	float* dev_sobelMaskX = 0;
	float* dev_sobelMaskY = 0;
	size_t pitchMask;
	
	cudaStatus = cudaMallocPitch((void**)&dev_sobelMaskX, &pitchMask, 3*sizeof(float), 3);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMallocPitch((void**)&dev_sobelMaskY, &pitchMask, 3*sizeof(float), 3);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy2D(dev_sobelMaskX, pitchMask, sobelMaskX, 3*sizeof(float), 3*sizeof(float), 3, cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }	

	cudaStatus = cudaMemcpy2D(dev_sobelMaskY, pitchMask, sobelMaskY, 3*sizeof(float), 3*sizeof(float), 3, cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	int numberThread = 32;
	int xDim = (width + numberThread - 1)/numberThread;
	int yDim = (height + numberThread - 1)/numberThread;

    // Launch a kernel on the GPU with one thread for each element.
//	localBinarization<<<dim3(xDim, yDim), dim3(numberThread, numberThread)>>>
//		(dev_dst, dev_src, dev_imgBorder, dev_sobelMaskX, dev_sobelMaskY, dev_theta, dev_sobel, width, height, pitch/sizeof(float), pitchMask/sizeof(float));

	sobel<<<dim3(xDim, yDim), dim3(numberThread, numberThread)>>>
		(dev_sobel, dev_src, width, height, dev_sobelMaskX, dev_sobelMaskY, dev_theta, pitch/sizeof(float), pitchMask/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	nonMaximumSuppression<<<dim3(xDim, yDim), dim3(numberThread, numberThread)>>>
		(dev_dst, dev_sobel, dev_theta, width, height, pitch/sizeof(float));
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	globalBinarisationInv<<<dim3(xDim, yDim), dim3(numberThread, numberThread)>>>
		(60, dev_imgBorder, dev_dst, width, height, pitch/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////
	int winWidth = 16;
	int winHeigth = 16;
	float scaleM = 1.3;

	int numberSubImgX = width/winWidth + (width%winWidth > 0 ? 1 : 0);
    int numberSubImgY = height/winHeigth + (height%winHeigth > 0 ? 1 : 0);

	int winWidScale = winWidth*scaleM;
	int winHeiScale = winHeigth*scaleM;

	float* dev_windows = 0;
	float* dev_winBorder = 0;
	float* dev_darkestColor = 0;
	float* dev_firstWin = 0;
	float* dev_secondWin = 0;
	size_t pitchWin;
	size_t pitchColor;
	//выделим память для всех окошек
	cudaStatus = cudaMallocPitch((void**)&dev_windows, &pitchWin, winWidScale*numberSubImgX*sizeof(float), winHeiScale*numberSubImgY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocPitch((void**)&dev_winBorder, &pitchWin, winWidScale*numberSubImgX*sizeof(float), winHeiScale*numberSubImgY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocPitch((void**)&dev_secondWin, &pitchWin, winWidScale*numberSubImgX*sizeof(float), winHeiScale*numberSubImgY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocPitch((void**)&dev_firstWin, &pitchWin, winWidScale*numberSubImgX*sizeof(float), winHeiScale*numberSubImgY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMallocPitch((void**)&dev_darkestColor, &pitchColor, numberSubImgX*sizeof(float), numberSubImgY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//
	dim3 threads = dim3(winWidth, winHeigth);
	dim3 blocks = dim3(numberSubImgX, numberSubImgY);

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	/////binaryzation
	emptyImg<<<blocks, dim3(winWidScale, winHeiScale)>>>(dev_winBorder, winWidScale*numberSubImgX, winHeiScale*numberSubImgY, pitchWin/sizeof(float));
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	copyImgToImgWin<<<blocks, threads>>>(dev_winBorder, dev_imgBorder, width, height, winWidth, winHeigth, winWidScale - winWidth, winHeiScale - winHeigth, pitch/sizeof(float), pitchWin/sizeof(float));
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	emptyImg<<<blocks, dim3(winWidScale, winHeiScale)>>>(dev_windows, winWidScale*numberSubImgX, winHeiScale*numberSubImgY, pitchWin/sizeof(float));
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	copyImgToImgWin<<<blocks, threads>>>(dev_windows, dev_src, width, height, winWidth, winHeigth, winWidScale - winWidth, winHeiScale - winHeigth, pitch/sizeof(float), pitchWin/sizeof(float));
	
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	initializationValue<<<dim3((numberSubImgX + 31)/32, (numberSubImgY + 31)/32 ),dim3(32,32)>>>(dev_darkestColor, -1, numberSubImgX, numberSubImgY, pitchColor/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	findMostDarkestColor<<<blocks, 1>>>(dev_darkestColor, dev_windows, dev_winBorder, winWidScale*numberSubImgX, winHeiScale*numberSubImgY, winWidScale, winHeiScale, pitchWin/sizeof(float), pitchColor/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

/*	float* img_dark = (float*) malloc(sizeof(float)*numberSubImgX*numberSubImgY);

	cudaStatus = cudaMemcpy2D(img_dark, numberSubImgX*sizeof(float), dev_darkestColor, pitchColor, numberSubImgX*sizeof(float), numberSubImgY, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	for(int i = 0; i < numberSubImgY; i++) {
		for(int j = 0; j < numberSubImgX; j++) {
			printf("%.0f ", img_dark[j + i*numberSubImgX]);
		}
		printf("\n");
	}

	free(img_dark);*/

	clarificationImg<<<blocks, dim3(winWidScale, winHeiScale)>>>(dev_firstWin, dev_windows, 153, winWidScale*numberSubImgX, winHeiScale*numberSubImgY, winWidScale, winHeiScale, pitchWin/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	fillingImg<<<blocks, dim3(winWidScale, winHeiScale)>>>(dev_secondWin, dev_windows, dev_darkestColor, winWidScale*numberSubImgX, winHeiScale*numberSubImgY, winWidScale, winHeiScale, pitchWin/sizeof(float), pitchColor/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	combinResBinImgs<<<blocks, dim3(winWidScale, winHeiScale)>>>(dev_dst, dev_firstWin, dev_secondWin, winWidScale*numberSubImgX, winHeiScale*numberSubImgY, winWidth,
		winHeigth, winWidScale - winWidth, winHeiScale - winHeigth, pitch/sizeof(float), pitchWin/sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy2D(img_dst, width*sizeof(float), dev_dst, pitch, width*sizeof(float), height, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	/*cudaStatus = cudaMemcpy2D(img_win, winWidScale*numberSubImgX*sizeof(float), dev_windows, pitchWin, winWidScale*numberSubImgX*sizeof(float), winHeiScale*numberSubImgY, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/

Error:
	cudaFree(dev_dst);
	cudaFree(dev_src); 
	cudaFree(dev_imgBorder);
	cudaFree(dev_sobelMaskX);
	cudaFree(dev_sobelMaskY);
	cudaFree(dev_theta);
	cudaFree(dev_sobel);
	cudaFree(dev_windows);
	cudaFree(dev_winBorder);
	cudaFree(dev_darkestColor);
	cudaFree(dev_firstWin);
	cudaFree(dev_secondWin);
    return cudaStatus;
}
