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

namespace AlgorithmVSCOME
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = "C:\\Users\\Tanya\\Documents\\tests_data\\101_s.tif";

            double[,] imgBytes = ImageHelper.LoadImage(path);
            imgBytes = ImageEnhancementHelper.EnhanceImage(imgBytes);

            double[,] orientationField = OrientationFieldGenerator.GenerateOrientationField(imgBytes.Select2D(x => (int)x));
            Complex[,] complexOrientationField = orientationField.Select2D(x => (new Complex(Math.Cos(2 * x), Math.Sin(2 * x))));

            Complex[,] filter = Filter.GetFilter(orientationField);

            Complex[,] complexFilteredField = ConvolutionHelper.ComplexConvolve(complexOrientationField, filter);
            double[,] filteredField = complexFilteredField.Select2D(x => x.Magnitude);

            ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\orientationField.jpg");
            ImageHelper.SaveArray(filteredField, "C:\\Users\\Tanya\\Documents\\Results\\filteredField.jpg");

            VSCOME vscome = new VSCOME(orientationField, filteredField);

            double[,] vscomeValue = vscome.CalculateVscomeValue();

            ImageHelper.SaveArray(vscomeValue, "C:\\Users\\Tanya\\Documents\\Results\\vscomeValue_1.jpg");

            Tuple<int, int> corePoint = KernelHelper.Max2dPosition(vscomeValue);

            Console.WriteLine("Reference point ({0},{1})", corePoint.Item1, corePoint.Item2);
            Console.ReadLine();
        }
    }
}
