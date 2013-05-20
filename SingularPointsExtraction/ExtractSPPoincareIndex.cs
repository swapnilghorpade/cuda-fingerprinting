using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FingerprintLib;
using FingerprintPhD.Common;
using ComplexFilterQA;

namespace SingularPointsExtraction
{
    class ExtractSPPoincareIndex
    {
        static public Tuple<int, int> ExtractSP(double[,] img)
        {
            double[,] directionField = PixelwiseOrientationFieldGenerator.GenerateOrientationField(img);
            double[,] squaredDirectionField = directionField.Select2D((x)=>(x*x));

            return null;
        }
        static private double[,] GenerateXGradients(double[,] source, double sigma)
        {
            double[,] kernelX = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, sigma) * x,
                                                  KernelHelper.GetKernelSizeForGaussianSigma(sigma));

            double[,] dx = ConvolutionHelper.Convolve(source, kernelX);
            return dx;
        }

        static private double[,] GenerateYGradients(double[,] source, double sigma)
        {
            double[,] kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, sigma) * x,
                                                  KernelHelper.GetKernelSizeForGaussianSigma(sigma));

            double[,] dy = ConvolutionHelper.Convolve(source, kernelY);
            return dy;
        }
    }
}
