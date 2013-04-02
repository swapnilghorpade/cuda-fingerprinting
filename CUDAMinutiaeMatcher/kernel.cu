#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "MinutiaMatching.h"
#include <time.h>
extern "C"{

__declspec(dllexport) int main();

}

const float tau1 = 0.35f;
const float tau2 = 0.45f;

int main()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);

	FillDirections();

	FILE* f = fopen("C:\\temp\\104_6.bin","rb");

	int width;
	int height;

	fread(&width,sizeof(int),1,f);
	fread(&height,sizeof(int),1,f);

	int* ar = (int*)malloc(sizeof(int)*width*height);
	float* ar2 = (float*)malloc(sizeof(float)*width*height);

	fread(ar,sizeof(int),width*height,f);
	for(int i=0;i<width*height;i++)
	{
		ar2[i]=ar[i];
	}
	fclose(f);
	
	CUDAArray<float> sourceImage = CUDAArray<float>(ar2,width,height);

	CUDAArray<float> g1 = Reduce(sourceImage,1.7f);
	CUDAArray<float> g2 = Reduce(g1,1.21f);
	CUDAArray<float> g3 = Reduce(g2,1.3f);
	CUDAArray<float> g4 = Reduce(g3,1.3f);

	CUDAArray<float> p3 = Expand(g4, 1.3f, g3.Width, g3.Height);
	CUDAArray<float> p2 = Expand(g3, 1.3f, g2.Width, g2.Height);
	CUDAArray<float> p1 = Expand(g2, 1.21f,g1.Width, g1.Height);

	
	SubtractArray(g3,p3);
	EnhanceContrast(g3);
	SubtractArray(g2,p2);
	EnhanceContrast(g2);
	SubtractArray(g1,p1);
	EnhanceContrast(g1);

	CUDAArray<float> ls1Real;
	CUDAArray<float> ls1Im;
	EstimateLS(&ls1Real, &ls1Im, g1, 0.9f, 1.5f);

	CUDAArray<float> ls2Real;
	CUDAArray<float> ls2Im;
	EstimateLS(&ls2Real, &ls2Im, g2, 0.9f, 3.2f);

	CUDAArray<float> ls3Real;
	CUDAArray<float> ls3Im;
	EstimateLS(&ls3Real, &ls3Im, g3, 0.9f, 3.2f);

	CorrectLS1WithLS2(ls1Real, ls1Im, ls2Real, ls2Im);

	DirectionFiltering(g1, ls1Real, ls1Im, tau1, tau2);
	DirectionFiltering(g2, ls2Real, ls2Im, tau1, tau2);
	DirectionFiltering(g3, ls3Real, ls3Im, tau1, tau2);

	CUDAArray<float> el3 = Expand(g3, 1.3f, g2.Width, g2.Height);
	AddArray(el3,g2);
	CUDAArray<float> el2 = Expand(el3, 1.21f,g1.Width, g1.Height);
	el3.Dispose();
	AddArray(el2,g1);
	CUDAArray<float> enhanced = Expand(el2, 1.7f,sourceImage.Width, sourceImage.Height);
	el2.Dispose();
	EnhanceContrast(enhanced);
	//SaveArray(enhanced, "C:\\temp\\104_6_enh.bin");
	//CUDAArray<float> magnitude = CUDAArray<float>(ls1Real.Width,ls1Real.Height);
	//	dim3 blockSize = dim3(defaultThreadCount,defaultThreadCount);
	//dim3 gridSize = 
	//	dim3(ceilMod(magnitude.Width, defaultThreadCount),
	//	ceilMod(magnitude.Height, defaultThreadCount));
	//cudaGetMagnitude<<<gridSize,blockSize>>>(magnitude, ls1Real, ls1Im);
	//SaveArray(magnitude,"C:\\temp\\104_6_mag3.bin");

	// minutia extraction
	clock_t clk1 = clock();
	CUDAArray<float> psReal;
	CUDAArray<float> psIm;
	CUDAArray<float> psM=CUDAArray<float>(enhanced.Width,enhanced.Height);
	CUDAArray<float> lsReal;
	CUDAArray<float> lsIm;
	CUDAArray<float> lsM=CUDAArray<float>(enhanced.Width,enhanced.Height);
	EstimateLS(&lsReal, &lsIm, enhanced, 0.9f, 1.5f);
	EstimatePS(&psReal, &psIm, enhanced, 0.9f, 2.5f);

	GetMagnitude(lsM, lsReal, lsIm);
	GetMagnitude(psM, psReal, psIm);

	CUDAArray<float> psi = CUDAArray<float>(lsM.Width, lsM.Height);
	EstimateMeasure(psi,lsM, psM);

	lsIm.Dispose();
	lsReal.Dispose();
	psIm.Dispose();
	psReal.Dispose();

	SaveArray(psi,"C:\\temp\\104_6_psi.bin");

	sourceImage.Dispose();
	enhanced.Dispose();
	g1.Dispose();
	g2.Dispose();
	g3.Dispose();
	g4.Dispose();
	p1.Dispose();
	p2.Dispose();
	p3.Dispose();
	free(ar);
	free(ar2);
	clock_t clk2 = clock();

	float dt = ((float)clk2-clk1)/ CLOCKS_PER_SEC;

	// minutia matching
	MatchFingers();
	cudaDeviceReset();
    return 0;
}