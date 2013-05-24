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
    class SPByPoincareIndex
    {
        static public Tuple<int, int> ExtractSP(double[,] img)
        {
            double[,] directionField = PixelwiseOrientationFieldGenerator.GenerateOrientationField(img);
            ImageHelper.SaveArray(directionField, "D:/img/poinc0.bmp");
            double[,] squaredDirectionField = directionField.Select2D((x)=>(x*x));
            ImageHelper.SaveArray(squaredDirectionField, "D:/img/poinc1.bmp");

            double[,] jx = GenerateXGradients(squaredDirectionField, 1d);
            ImageHelper.SaveArray(jx, "D:/img/poinc2x.bmp");
            double[,] jy = GenerateYGradients(squaredDirectionField, 1d);
            ImageHelper.SaveArray(jy, "D:/img/poinc2y.bmp");

            //почему-то получаются одинаковыми?
            double[,] jxdy = GenerateYGradients(jx, 0.8);
            ImageHelper.SaveArray(jxdy, "D:/img/poinc3xy.bmp");
            double[,] jydx = GenerateXGradients(jy, 0.8);
            ImageHelper.SaveArray(jydx, "D:/img/poinc3yx.bmp");
            //а тут соответственно нули             
            double[,] result = jydx.Select2D((a,x,y)=>(jydx[x,y] - jxdy[x,y]));

            double max = KernelHelper.Max2d(result.Select2D((x)=>Math.Abs(x)));
            ImageHelper.SaveArray(result, "D:/img/poinc.bmp");

            Tuple<int, int> pointMax = KernelHelper.Max2dPosition(result);

            return pointMax;
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
            double[,] kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, sigma) * (-y),
                                                  KernelHelper.GetKernelSizeForGaussianSigma(sigma));

            double[,] dy = ConvolutionHelper.Convolve(source, kernelY);
            return dy;
        }
    }
}
