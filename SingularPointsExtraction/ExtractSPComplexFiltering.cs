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
            ImageHelper.SaveArray(level1, "D:/level1.bmp");
            ImageHelper.SaveArray(level2, "D:/level2.bmp");
            ImageHelper.SaveArray(level3, "D:/level3.bmp");

            //Tuple<int,int> point3 = CalculateFilterResponse(level3, 0.6, 1.5, true);

            //double[,] windowFor2 = GetWindowForSearch(13, level2, point3);
            //Tuple<int, int> point2 = CalculateFilterResponse(windowFor2, 0.6, 1.5, false);


            //double[,] windowFor1 = GetWindowForSearch(13, level1, point2);
            //Tuple<int, int> point1 = CalculateFilterResponse(windowFor1, 0.6, 1.5, false);


            //double[,] windowFor0= GetWindowForSearch(13, level0, point1);
            //Tuple<int, int> point0 = CalculateFilterResponse(windowFor0, 0.6, 1.5, false);

            Tuple<int,int> point0 = CalculateFilterResponse(level2, 0.6, 1.5, true);

            return point0;
        }

        static private Tuple<int, int> CalculateFilterResponse(double[,] img, double sigma1, double sigma2, bool isLevel3)
        {
            Complex[,] response = SymmetryHelper.EstimatePS(img, sigma1, sigma2);
            double[,] aaa1 = response.Select2D((value, row, column) => (response[row, column].Magnitude));
            ImageHelper.SaveArray(aaa1, "D:/response.bmp");

            Complex[,] responseForDelta = SymmetryHelper.EsimateH2(img, sigma1, sigma2);
            double[,] aaa2 = responseForDelta.Select2D((value, row, column) => (responseForDelta[row, column].Magnitude));
            ImageHelper.SaveArray(aaa1, "D:/responseForDelta.bmp");

            for (int i = 0; i < response.GetLength(0); i++)
            {
                for (int j = 0; j < response.GetLength(1); j++)
                {
                    double newMagnitude = response[i, j].Magnitude * (1d - responseForDelta[i, j].Magnitude);
                    response[i, j] = Complex.FromPolarCoordinates(newMagnitude, response[i, j].Phase);
                }
            }
            double[,] aaa = response.Select2D((value, row, column) => (response[row, column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/resp12.bmp");

            if (isLevel3)
            {
                response = CalculateModifiedFilterResponse(img, response, sigma1);
            }
            
            //тут будет поиск абсолютного максимума
            return new Tuple<int, int>(0, 0);
        }

        static private Complex[,] CalculateModifiedFilterResponse(double[,] img, Complex[,] response, double sigma1)
        {
            Complex[,] z = SymmetryHelper.GetSquaredDerectionField(img, sigma1);
            double[,] magnitudeZ = z.Select2D((value, x, y) => (z[x, y].Magnitude));

            ImageHelper.SaveArray(magnitudeZ, "D:/magnitudeZ.bmp");
            double[,] gaussians = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, 1.5d), KernelHelper.GetKernelSizeForGaussianSigma(1.5d));
            double[,] resultOfconvolve = ConvolutionHelper.Convolve(magnitudeZ, gaussians);
            Complex[,] zc = response.Select2D((value, x, y) => (response[x, y] * resultOfconvolve[x, y]));

            double[,] bigGaussian = MakeGaussian((x, y) =>
                Gaussian.Gaussian2D((float)x, (float)y, 8, 5), img.GetLength(0), img.GetLength(1));
            Complex[,] gc = response.Select2D((value, x, y) => (response[x, y] * bigGaussian[x, y]));

            Complex[,] modifiedResponse = gc.Select2D((value, x, y) => 0.5 * (gc[x, y] + zc[x, y]));


            double[,] aaa = modifiedResponse.Select2D((value, row, column) => (modifiedResponse[row, column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/resp12M.bmp");

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

        private static double[,] GetWindowForSearch(int size, double[,] img, Tuple<int,int> point)
        {

            int a = (point.Item1 <= size / 2)? (size/2) : point.Item1;
            int b = (point.Item2 <= size / 2) ? (size / 2) : point.Item2;
            a = (point.Item1 >= img.GetLength(0) - (size/2)-1)? img.GetLength(0) - (size/2)-1 : a;
            b = (point.Item2 >= img.GetLength(1) - (size/2)-1)? img.GetLength(1) - (size/2)-1 : b;
            a = a - size / 2;
            b = b - size / 2;
            double[,] result = new double[size,size];
            result = result.Select2D((c,x,y)=>(img[a+x,b+y]));
            return result;
        }

    }
}

