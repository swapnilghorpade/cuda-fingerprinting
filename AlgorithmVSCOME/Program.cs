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

            for (int i = 0; i < 1 /*pathes.GetLength(0)*/; i++)
            {
                Tuple<int, int> redPoint = ImageHelper.FindRedPoint(pathes[i]);
                double[,] imgBytes = ImageEnhancementHelper.EnhanceImage(ImageHelper.LoadImage(pathes[i]));

                double[,] orientationField = OrientationFieldGenerator.GenerateOrientationField(imgBytes.Select2D(x => (int)x));
                Complex[,] complexOrientationField = orientationField.Select2D(x => (new Complex(Math.Cos(2 * x), Math.Sin(2 * x))));

                Complex[,] filter = Filter.GetFilter(orientationField);
                Complex[,] complexFilteredField = ConvolutionHelper.ComplexConvolve(complexOrientationField, filter);
                double[,] filteredField = complexFilteredField.Select2D(x => x.Magnitude);

                var kernelSymmetry = KernelHelper.MakeComplexKernel(
                (x, y) =>
                y < 0
                    ? 0
                    : ComplexFilterQA.Gaussian.Gaussian2D(x, y, 1.5) * y * x * 2.0d /
                      (x == 0 && y == 0 ? 1 : Math.Sqrt(x * x + y * y)),
                (x, y) =>
                y < 0
                    ? 0
                    : ComplexFilterQA.Gaussian.Gaussian2D(x, y, 1.5) * (x * x - y * y) /
                      (x == 0 && y == 0 ? 1 : Math.Sqrt(x * x + y * y)),
                KernelHelper.GetKernelSizeForGaussianSigma(Constants.Sigma));

                Complex[,] complexSymmetry = ConvolutionHelper.ComplexConvolve(complexOrientationField, kernelSymmetry);
                double[,] symmetry = complexSymmetry.Select2D(x => x.Magnitude);
                double[,] vscomeValue = VSCOME.CalculateVscomeValue(orientationField, filteredField, symmetry);
                Tuple<int, int> corePoint = KernelHelper.Max2dPosition(vscomeValue);

                writer.WriteLine(GetDistance(redPoint, corePoint));

                // ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\orientationField.jpg");
                // ImageHelper.SaveArray(filteredField, "C:\\Users\\Tanya\\Documents\\Results\\filteredField.jpg");
                //ImageHelper.SaveArray(vscomeValue, "C:\\Users\\Tanya\\Documents\\Results\\vscomeValue.jpg");
            }

            writer.Close();
        }

        private static double GetDistance(Tuple<int, int> a, Tuple<int, int> b)
        {
            return Math.Sqrt(Math.Pow((a.Item1 - b.Item1), 2) + Math.Pow((a.Item2 - b.Item2), 2));
        }
    }
}
