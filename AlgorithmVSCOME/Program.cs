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
using FingerprintPhD.Common;
using System.IO;

namespace AlgorithmVSCOME
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] pathes = Directory.GetFiles("C:\\Users\\Tanya\\Documents\\tests_data\\db");
            StreamWriter writer = new StreamWriter("C:\\Users\\Tanya\\Documents\\Results\\AlgorithmVSCOMEResult.txt", true);

            for (int i = 50; i < 60 /*pathes.GetLength(0)*/; i++)
            {
                Tuple<int, int> redPoint = ImageHelper.FindRedPoint(pathes[i]);
                double[,] imgBytes = ImageEnhancementHelper.EnhanceImage(ImageHelper.LoadImage(pathes[i]));
                double[,] orientationField = OrientationFieldGenerator.GenerateOrientationField(imgBytes.Select2D(x => (int)x));
                Complex[,] complexOrientationField = orientationField.Select2D(x => (new Complex(Math.Cos(2 * x), Math.Sin(2 * x))));
                
                Complex[,] filter = Filter.GetFilter(orientationField);
                Complex[,] complexFilteredField = ConvolutionHelper.ComplexConvolve(complexOrientationField, filter);
                double[,] filteredField = complexFilteredField.Select2D(x => x.Magnitude);
                
                double[,] vscomeValue = 
                    VSCOME.CalculateVscomeValue(orientationField, filteredField, Symmetry.GetSymmetry(complexOrientationField));
                Tuple<int, int> corePoint = StretchCoordinates(KernelHelper.Max2dPosition(vscomeValue), imgBytes, orientationField);

                writer.WriteLine(GetDistance(redPoint, corePoint));
            }

            writer.Close();
        }

        private static Tuple<int, int> StretchCoordinates(Tuple<int, int> point, double[,] imgBytes, double[,] orientationField)
        {
            int x = point.Item1 * (imgBytes.GetLength(0) / orientationField.GetLength(0));
            int y = point.Item2 * (imgBytes.GetLength(1) / orientationField.GetLength(1));

            return new Tuple<int, int>(x, y);
        }

        private static double GetDistance(Tuple<int, int> a, Tuple<int, int> b)
        {
            return Math.Sqrt(Math.Pow((a.Item1 - b.Item1), 2) + Math.Pow((a.Item2 - b.Item2), 2));
        }
    }
}
