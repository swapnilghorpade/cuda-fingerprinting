using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using ComplexFilterQA;
using FingerprintPhD;
using FingerprintLib;

namespace AlgorithmVSCOME
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = "C:\\Users\\Tanya\\Documents\\tests_data\\101_1.tif";

            double[,] imgBytes = ImageHelper.LoadImage(path);
            imgBytes = ImageEnhancementHelper.EnhanceImage(imgBytes);

            double[,] orientationField = OrientationFieldGenerator.GenerateOrientationField(imgBytes.Select2D((value, x, y) => ((int)imgBytes[x, y])));
            Complex[,] complexOrientationField = orientationField.Select2D((value, x, y) => 
                (new Complex(Math.Cos(2 * orientationField[x, y]), Math.Sin(2 * orientationField[x, y]))));

            Complex[,] filter = Filter.GetFilter(orientationField);

            Complex[,] complexFilteredField = ConvolutionHelper.ComplexConvolve(complexOrientationField, filter);
            double[,] filteredField = complexFilteredField.Select2D((value, x, y) => (complexFilteredField[x, y].Magnitude));

           // ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\orientationField.jpg");
           // ImageHelper.SaveArray(filteredField, "C:\\Users\\Tanya\\Documents\\Results\\filteredField.jpg"); 

            VSCOME vscome = new VSCOME(orientationField, filteredField);

            
            double[,] vscomeValue = vscome.CalculateVscomeValue();

           // ImageHelper.SaveArray(vscomeValue, "C:\\Users\\Tanya\\Documents\\Results\\vscomeValue.jpg"); 

            int xCoordinate = 0;
            int yCoordinate = 0;
            double max = Max2d(vscomeValue, ref xCoordinate, ref yCoordinate);

            Console.WriteLine("Reference point ({0},{1})", xCoordinate, yCoordinate);
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
    }
}
