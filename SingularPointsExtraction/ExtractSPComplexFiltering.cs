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

            double[,] level1 = ImageSizeHelper.Reduce2(level0, 2d);
            double[,] level2 = ImageSizeHelper.Reduce2(level1, 2d);
            double[,] level3 = ImageSizeHelper.Reduce2(level2, 2d);
            List<double[,]> gaussianPyramid = new List<double[,]> { level3, level2, level1, level0 };
            //ImageHelper.SaveArray(level1, "D:/1-1.bmp");
            ImageHelper.SaveArray(level2, "D:/level2.bmp");
            ImageHelper.SaveArray(level3, "D:/level3.bmp");


            //Calculate response of parabolic symmetry filter



            CalculateFilterResponse(level2, 0.6, 1.5, true);
            //Compute a modified complex filter response(for level 3)

            return null;
        }

        /// <summary>
        ///Calculate response of parabolic symmetry filter
        /// </summary>
        /// <param name="img"></param>
        /// <param name="sigma1">for direction field</param>
        /// <param name="sigma2">for direction field</param>
        /// <param name="isLevel3">is level 3 of gaussian pyramid</param>
        /// <returns></returns>
        static private Tuple<int, int> CalculateFilterResponse(double[,] img, double sigma1, double sigma2, bool isLevel3)
        {
            //double[,] forDelta3 = img.Select2D((value, row, column) => (img[row, img.GetLength(1) - column - 1]));

            Complex[,] response = SymmetryHelper.EstimatePS(img, sigma1, sigma2);
            //Complex[,] responseForDelta = SymmetryHelper.EstimatePS(forDelta3, sigma1, sigma2);
            //for (int i = 0; i < response.GetLength(0); i++)
            //{
            //    for (int j = 0; j < response.GetLength(1); j++)
            //    {
            //        double newMagnitude = response[i, j].Magnitude * (1d - responseForDelta[i, j].Magnitude);
            //        response[i, j] = Complex.FromPolarCoordinates(newMagnitude, response[i, j].Phase);
            //    }
            //}
            double[,] aaa = response.Select2D((value, row, column) => (response[row, column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/1-5.bmp");

            if (isLevel3)
            {
                response = CalculateModifiedFilterResponse(img, response, sigma1);
            }
            //тут будет поиск абсолютного максимума
            return new Tuple<int, int>(0, 0);
        }

        static private Complex[,] CalculateModifiedFilterResponse(double[,] img, Complex[,] response, double sigma1)
        {
            Complex[,] z = GetZ(img, sigma1);
            double[,] magnitudeZ = z.Select2D((value, x, y) => (z[x, y].Magnitude));

            ImageHelper.SaveArray(magnitudeZ, "D:/1-3.bmp");
            double[,] gaussians = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, 1.5d), KernelHelper.GetKernelSizeForGaussianSigma(1.5d));
            double[,] resultOfconvolve = ConvolutionHelper.Convolve(magnitudeZ, gaussians);
            Complex[,] zc = response.Select2D((value, x, y) => (response[x, y] * resultOfconvolve[x, y]));

            double[,] bigGaussian = MakeGaussian((x, y) => Gaussian.Gaussian2D((float)x, (float)y, 8, 5), img.GetLength
            (0), img.GetLength(1));
            Complex[,] gc = response.Select2D((value, x, y) => (response[x, y] * bigGaussian[x, y]));

            Complex[,] modifiedResponse = gc.Select2D((value, x, y) => 0.5 * (gc[x, y] + zc[x, y]));


            double[,] aaa = modifiedResponse.Select2D((value, row, column) => (modifiedResponse[row, column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/1-6.bmp");

            return modifiedResponse;

        }

        static double[,] MakeGaussian(Func<float, float, double> function, int sizeX, int sizeY)
        {
            double[,] kernel = new double[sizeX, sizeY];
            int centerX = sizeX / 2;
            int centerY = sizeY / 2;
            float shiftX = (sizeX - 2 * centerX - 1) / 2f;
            float shiftY = (sizeY - 2 * centerY - 1) / 2f;

            for (int x = -centerX; x <= centerX + 2 * shiftX; x++)
            {
                for (int y = -centerY; y <= centerY + 2 * shiftY; y++)
                {
                    kernel[centerX + x, centerY + y] = function(x - shiftX, y - shiftY);
                }
            }
            return kernel;
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
