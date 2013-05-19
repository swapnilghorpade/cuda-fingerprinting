using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using ComplexFilterQA;

namespace SingularPointsExtraction
{
    class SPByComplexFiltering
    {
        static public Tuple<int, int> ExtractSP(double[,] img)
        {
            //make Gaussian pyramid
            double[,] level0 = img;
            double[,] level1 = ChangingSize.Reduce2(level0, 2d);
            double[,] level2 = ChangingSize.Reduce2(level1, 2d);
            double[,] level3 = ChangingSize.Reduce2(level2, 2d);

            //ImageHelper.SaveArray(level1, "D:/1-1.bmp");
            //ImageHelper.SaveArray(level2, "D:/1-2.bmp");
            //ImageHelper.SaveArray(level3, "D:/1-3.bmp");

            //Calculate response of parabolic symmetry filter

            double[,] forDelta3 = level3.Select2D((value, row, column) => (level3[row, level3.GetLength(1) - column - 1]));
  
            Complex[,] response3 = Symmetry.EstimatePS(level3, 1.6d, 3.2d);                      
            Complex[,] responseForDelta3 = Symmetry.EstimatePS(forDelta3, 0.6d, 3.2d);
            for (int i = 0; i < response3.GetLength(0); i++)
            {
                for (int j = 0; j < response3.GetLength(1); j++)
                {
                    double newMagnitude = response3[i, j].Magnitude * (1d - responseForDelta3[i, j].Magnitude);
                    response3[i, j] = Complex.FromPolarCoordinates(newMagnitude, response3[i,j].Phase);                    
                }                
            }

            double[,] bbb = response3.Select2D((value, row, column) => (response3[row, column].Magnitude));
            ImageHelper.SaveArray(bbb, "D:/1-4.bmp");

            //Compute a modified complex filter response(for level 3)
            Complex[,] z = GetZ(level3, 1.6d);
            double[,] magnitudeZ = z.Select2D((value,x,y)=>(z[x,y].Magnitude));

            ImageHelper.SaveArray(magnitudeZ, "D:/1-3.bmp");
            double[,] gaussians = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, 1.5d), KernelHelper.GetKernelSizeForGaussianSigma(1.5d));
            double[,] resultOfconvolve = ConvolutionHelper.Convolve(magnitudeZ, gaussians);
            Complex[,] zc = response3.Select2D((value,x,y)=>(response3[x,y]*resultOfconvolve[x,y]));

            double[,] bigGaussian = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D((float)x, (float)y, 11.7, 7), response3.GetLength(0), response3.GetLength(1));
            Complex[,] gc = response3.Select2D((value, x, y) => (response3[x, y] * bigGaussian[x, y]));

            Complex[,] modifiedResponse3 = gc.Select2D((value, x, y) => 0.5*(gc[x, y] + zc[x, y]));

            
            double[,] aaa = modifiedResponse3.Select2D((value, row, column) => (modifiedResponse3[row,column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/1-5.bmp");
            
            return null;
        }

        //part of Symetry.EstimatePS
        //z - (direction field)^2
        static private Complex[,] GetZ(double[,] l1, double Sigma1)
        {
            var kernelX = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma1) * x, KernelHelper.GetKernelSizeForGaussianSigma(Sigma1));
            var resultX = ConvolutionHelper.Convolve(l1, kernelX);
            var kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma1) * -y, KernelHelper.GetKernelSizeForGaussianSigma(Sigma1));
            var resultY = ConvolutionHelper.Convolve(l1, kernelY);

            var preZ = KernelHelper.MakeComplexFromDouble(resultX, resultY);

            var z = preZ.Select2D(x => x * x);
            return z;
        }
    }
}
