using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using ComplexFilterQA;
using FingerprintLib;

namespace AlgorithmVSCOME
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Write path, please.");

            string path = Console.ReadLine();

            double[,] imgBytes = ImageHelper.LoadImage(path);
            imgBytes = ImageEnhancementHelper.EnhanceImage(imgBytes);

            double[,] orientationField = OrientationFieldGenerator.GenerateOrientationField(
                                                            DoubleArrayToIntArray(imgBytes));
            Complex[,] complexOrientationField = CalculateComplexOrientationField(orientationField);            
            Complex[,] filter = Filter.GetFilter(imgBytes);

            VSCOME vscome = new VSCOME(orientationField,
                ComplexArrayToDoubleArray(ConvolutionHelper.ComplexConvolve(complexOrientationField, filter)));

            double[,] vscomeValue = vscome.CalculateVscomeValue();
            int x = 0;
            int y = 0;
            double max = Max2d(vscomeValue, ref x, ref y);

            // (x, y) - finally located block
            // the position of the reference point is the center pixel of the finally located block
        }

        private static Complex[,] CalculateComplexOrientationField(double[,] orientationField)
        {
            int xLength = orientationField.GetLength(0);
            int yLength = orientationField.GetLength(1);
            Complex[,] result = new Complex[xLength, yLength];
            double doubleAngle = 0;

            for (int x = 0; x < xLength; x++)
            {
                for (int y = 0; y < yLength; y++)
                {
                    doubleAngle = 2 * orientationField[x,y];
                    result[x, y] = new Complex(Math.Cos(doubleAngle), Math.Sin(doubleAngle));
                }
            }

            return result;
        }

        private static double Max2d(double[,] arr, ref int xPosition, ref int yPosition)
        {
            double max = double.NegativeInfinity;
            for (int x = 0; x < arr.GetLength(0); x++)
            {
                for (int y = 0; y < arr.GetLength(1); y++)
                {
                    if (arr[x, y] > max)
                    {
                        max = arr[x, y];
                        xPosition = x;
                        yPosition = y;
                    }
                }
            }
            return max;
        }

        private static int[,] DoubleArrayToIntArray(double[,] arr)
        {
            int iLength = arr.GetLength(0);
            int jLength = arr.GetLength(1);
            int[,] result = new int[iLength, jLength];

            for (int i = 0; i < iLength; i++)
            {
                for (int j = 0; j < jLength; j++)
                {
                    result[i, j] = (int)arr[i, j];
                }
            }

            return result;
        }


        private static double[,] ComplexArrayToDoubleArray(Complex[,] arr)
        {
            int iLength = arr.GetLength(0);
            int jLength = arr.GetLength(1);
            double[,] result = new double[iLength, jLength];

            for (int i = 0; i < iLength; i++)
            {
                for (int j = 0; j < jLength; j++)
                {
                    result[i, j] = Complex.Abs(arr[i, j]);
                }
            }

            return result;
        }
    }
}
