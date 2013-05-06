using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

namespace ComplexFilterQA
{
    public class KernelHelper
    {
        public static int GetKernelSizeForGaussianSigma(double sigma)
        {
            return 2*(int) Math.Ceiling(sigma*3.0f) + 1;
        }

        public static Complex[,] MakeComplexKernel(Func<int, int, double> realFunction,
                                                   Func<int, int, double> imaginaryFunction, int size)
        {
            var realPart = MakeKernel(realFunction, size);
            var imPart = MakeKernel(imaginaryFunction, size);
            return MakeComplexFromDouble(realPart, imPart);
        }

        public static double Max2d(double[,] arr)
        {
            double max = double.NegativeInfinity;
            for (int x = 0; x < arr.GetLength(0); x++)
            {
                for (int y = 0; y < arr.GetLength(1); y++)
                {
                    if (arr[x, y] > max) max = arr[x, y];
                }
            }
            return max;
        }

        public static double[,] MakeKernel(Func<int, int, double> function, int size)
        {
            double[,] kernel = new double[size,size];
            int center = size/2;
            double sum = 0;
            for (int x = -center; x <= center; x++)
            {
                for (int y = -center; y <= center; y++)
                {
                    sum += kernel[center + x, center + y] = function(x, y);
                }
            }
            // normalization
            if (Math.Abs(sum) >0.0000001)
                for (int x = -center; x <= center; x++)
                {
                    for (int y = -center; y <= center; y++)
                    {
                        kernel[center + x, center + y] /= sum;
                    }
                }
            return kernel;
        }

        public static double[,] GetRealPart(Complex[,] array)
        {
            double[,] result = new double[array.GetLength(0),array.GetLength(1)];
            for (int x = 0; x < array.GetLength(0); x++)
            {
                for (int y = 0; y < array.GetLength(1); y++)
                {
                    result[x, y] = array[x, y].Real;
                }
            }
            return result;
        }

        public static double[,] GetMagnitude(Complex[,] array)
        {
            double[,] result = new double[array.GetLength(0),array.GetLength(1)];
            for (int x = 0; x < array.GetLength(0); x++)
            {
                for (int y = 0; y < array.GetLength(1); y++)
                {
                    result[x, y] = array[x, y].Magnitude;
                }
            }
            return result;
        }

        public static double[,] GetPhase(Complex[,] array)
        {
            double[,] result = new double[array.GetLength(0),array.GetLength(1)];
            for (int x = 0; x < array.GetLength(0); x++)
            {
                for (int y = 0; y < array.GetLength(1); y++)
                {
                    result[x, y] = array[x, y].Phase;
                }
            }
            return result;
        }

        public static double[,] GetImaginaryPart(Complex[,] array)
        {
            double[,] result = new double[array.GetLength(0),array.GetLength(1)];
            for (int x = 0; x < array.GetLength(0); x++)
            {
                for (int y = 0; y < array.GetLength(1); y++)
                {
                    result[x, y] = array[x, y].Imaginary;
                }
            }
            return result;
        }

        public static Complex[,] MakeComplexFromPolar(double[,] magnitude, double[,] phase)
        {
            int maxX = magnitude.GetLength(0);
            int maxY = magnitude.GetLength(1);
            Complex[,] result = new Complex[maxX, maxY];
            for (int x = 0; x < maxX; x++)
            {
                for (int y = 0; y < maxY; y++)
                {
                    result[x, y] = Complex.FromPolarCoordinates(magnitude[x, y], phase[x, y]);
                }
            }
            return result;
        }

    public static Complex[,] MakeComplexFromDouble(double[,] real, double[,] imaginary)
        {
            int maxX = real.GetLength(0);
            int maxY = real.GetLength(1);
            Complex[,] result = new Complex[maxX, maxY];
            for (int x = 0; x <maxX; x++)
            {
                for (int y = 0; y <maxY; y++)
                {
                    result[x, y] = new Complex(real[x,y],imaginary[x,y]);
                }
            }
            return result;
        }

        public static double[,] Subtract(double[,] source, double[,] value)
        {
            var maxX = source.GetLength(0);
            var maxY = source.GetLength(1);
            var result = new double[maxX, maxY];
            for (int x = 0; x < maxX; x++)
            {
                for (int y = 0; y < maxY; y++)
                {
                    result[x, y] = source[x, y] - value[x, y];
                }
            }
            return result;
        }

        public static double[,] Zip2D(double[,] arr1, double[,] arr2, Func<double,double,double> f)
        {
            var result = new double[arr1.GetLength(0), arr1.GetLength(1)];
            for (int x = 0; x < arr1.GetLength(0); x++)
            {
                for (int y = 0; y < arr1.GetLength(1); y++)
                {
                    result[x, y] = f(arr1[x, y], arr2[x, y]);
                }
            }
            return result;
        }

        public static double[,] Add(double[,] source, double[,] value)
        {
            var maxX = source.GetLength(0);
            var maxY = source.GetLength(1);
            var result = new double[maxX, maxY];
            for (int x = 0; x < maxX; x++)
            {
                for (int y = 0; y < maxY; y++)
                {
                    result[x, y] = source[x, y] + value[x, y];
                }
            }
            return result;
        }
    }
}
