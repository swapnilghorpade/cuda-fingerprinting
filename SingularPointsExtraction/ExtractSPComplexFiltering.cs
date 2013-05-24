using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
//using System.Drawing.Imaging;
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

            ImageHelper.SaveArray(level1, "D:/img/level1.bmp");
            ImageHelper.SaveArray(level2, "D:/img/level2.bmp");
            ImageHelper.SaveArray(level3, "D:/img/level3.bmp");

            double[,] response3 = CalculateFilterResponse(level3, 0.6, 1.5, true);
            double[,] response2 = CalculateFilterResponse(level2, 0.6, 1.5, true);
            double[,] response1 = CalculateFilterResponse(level1, 0.6, 1.5, false);
            double[,] response0 = CalculateFilterResponse(level0, 0.6, 1.5, false);

            ImageHelper.SaveArray(response3, "D:/img/response3.bmp");
            ImageHelper.SaveArray(response2, "D:/img/response2.bmp");
            ImageHelper.SaveArray(response1, "D:/img/response1.bmp");
            ImageHelper.SaveArray(response0, "D:/img/response0.bmp");
            //Tuple<int, int> point3 = FindPoint(response3);
            //Tuple<int, int> point2 = FindSecondPoint(response2, point3);

            Tuple<int, int> point2 = FindPoint(response2);
            Tuple<int, int> point1 = FindSecondPoint(response1, point2);
            Tuple<int, int> point0 = FindSecondPoint(response0, point1);

            return point0;
        }

        private static Tuple<int, int> FindPoint(double[,] response)
        {
            double[,] responseWithoutBound = new double[response.GetLength(0) - 10, response.GetLength(0) - 10];
            responseWithoutBound = responseWithoutBound.Select2D((a, x, y) => (response[x + 5, y + 5]));

            Tuple<int, int> point = KernelHelper.Max2dPosition(responseWithoutBound);

            return new Tuple<int, int>(point.Item1 + 5, point.Item2 + 5);
        }

        private static Tuple<int, int> FindSecondPoint(double[,] response, Tuple<int, int> prevPoint)
        {
            Tuple<int, int> centerOfWindow = new Tuple<int, int>(prevPoint.Item1 * 2, prevPoint.Item2 * 2);

            int a = (centerOfWindow.Item1 <= 10) ? (centerOfWindow.Item1 + 1) : centerOfWindow.Item1;
            int b = (centerOfWindow.Item2 <= 10) ? (centerOfWindow.Item2 + 1) : centerOfWindow.Item2;
            a = (centerOfWindow.Item1 >= response.GetLength(0) - 11) ? centerOfWindow.Item1 - 1 : a;
            b = (centerOfWindow.Item2 >= response.GetLength(1) - 11) ? centerOfWindow.Item2 - 1 : b;

            centerOfWindow = new Tuple<int, int>(a, b);

            double[,] window = GetWindowForSearch(13, response, centerOfWindow);

            Tuple<int, int> newPoint = KernelHelper.Max2dPosition(window);

            newPoint = new Tuple<int, int>(newPoint.Item1 + centerOfWindow.Item1 - 6, newPoint.Item2 + centerOfWindow.Item2 - 6);

            return newPoint;            
        }

        static private double[,] CalculateFilterResponse(double[,] img, double sigma1, double sigma2, bool isLevel3)
        {
            Complex[,] response = SymmetryHelper.EstimatePS(img, sigma1, sigma2);
            double[,] aaa1 = response.Select2D((value, row, column) => (response[row, column].Magnitude));
            ImageHelper.SaveArray(aaa1, "D:/img/response.bmp");

            Complex[,] responseForDelta = SymmetryHelper.EsimateH2(img, sigma1, sigma2);
            double[,] aaa2 = responseForDelta.Select2D((value, row, column) => (responseForDelta[row, column].Magnitude));
            ImageHelper.SaveArray(aaa1, "D:/img/responseForDelta.bmp");

            for (int i = 0; i < response.GetLength(0); i++)
            {
                for (int j = 0; j < response.GetLength(1); j++)
                {
                    double newMagnitude = response[i, j].Magnitude * (1d - responseForDelta[i, j].Magnitude);
                    response[i, j] = Complex.FromPolarCoordinates(newMagnitude, response[i, j].Phase);
                }
            }
            double[,] aaa = response.Select2D((value, row, column) => (response[row, column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/img/resp12.bmp");

            if (isLevel3)
            {
                response = CalculateModifiedFilterResponse(img, response, sigma1);
            }

            double[,] magnitudeOfResponse = response.Select2D((x)=>x.Magnitude);

            return magnitudeOfResponse;
        }

        static private Complex[,] CalculateModifiedFilterResponse(double[,] img, Complex[,] response, double sigma1)
        {
            Complex[,] z = SymmetryHelper.GetSquaredDerectionField(img, sigma1);
            double[,] magnitudeZ = z.Select2D((value, x, y) => (z[x, y].Magnitude));

            ImageHelper.SaveArray(magnitudeZ, "D:/img/magnitudeZ.bmp");
            double[,] gaussians = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, 1.5d), KernelHelper.GetKernelSizeForGaussianSigma(1.5d));
            double[,] resultOfconvolve = ConvolutionHelper.Convolve(magnitudeZ, gaussians);
            Complex[,] zc = response.Select2D((value, x, y) => (response[x, y] * resultOfconvolve[x, y]));

            double[,] bigGaussian = MakeGaussian((x, y) =>
                Gaussian.Gaussian2D((float)x, (float)y, 8, 5), img.GetLength(0), img.GetLength(1));
            Complex[,] gc = response.Select2D((value, x, y) => (response[x, y] * bigGaussian[x, y]));

            Complex[,] modifiedResponse = gc.Select2D((value, x, y) => 0.5 * (gc[x, y] + zc[x, y]));


            double[,] aaa = modifiedResponse.Select2D((value, row, column) => (modifiedResponse[row, column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/img/resp12M.bmp");

            
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

