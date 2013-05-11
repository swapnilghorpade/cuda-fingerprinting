using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
            // imgBytes = ComplexFilterQA.EnhanceImage(imgBytes);

            /* double[,] imgBytes
               but 
               OrientationFieldGenerator.GenerateOrientationField(int[,] imgBytes)
            */
            
            // double[,] orientationField = OrientationFieldGenerator.GenerateOrientationField(imgBytes);
            double[,] orientationField = new double[1,1];

            VSCOME vscome = new VSCOME(orientationField);

            double[,] vscomeValue = vscome.CalculateVscomeValue();

            int x = 0;
            int y = 0;
            double max = Max2d(vscomeValue, ref x, ref y);

            // (x, y) - finally located block
            // the position of the reference point is the center pixel of the finally located block
        }

        internal static double Max2d(double[,] arr, ref int xPosition, ref int yPosition)
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
    }
}
