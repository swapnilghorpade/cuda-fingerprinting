#include "cuda_runtime.h"
#include "math_functions.h"
#include "math_constants.h"
#include "math.h"

__device__ float Gaussian2D(float x, float y, float sigma)
{
	float commonDenom = sigma*sigma*2.0f;
	float denominator = commonDenom*CUDART_PI_F;
	return expf(-(x*x+y*y)/commonDenom)/denominator;
}

__device__ float Gaussian1D(float x, float sigma)
{
	float commonDenom = sigma*sigma*2.0f;
	float denominator = sigma*sqrtf(CUDART_PI_F*2.0f);
	return expf(-(x*x)/commonDenom)/denominator;
}