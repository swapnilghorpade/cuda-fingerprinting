using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;
using FingerprintLib;

namespace ModelBasedAlgorithm
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = "C:\\Users\\Tanya\\Documents\\tests_data\\101_1.tif";

            double[,] imgBytes = ImageHelper.LoadImage(path);
            imgBytes = ImageEnhancementHelper.EnhanceImage(imgBytes);

            double[,] orientationField = PixelwiseOrientationFieldGenerator.GenerateOrientationField(imgBytes);

            ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\china\\orientationField.jpg");

            List<Tuple<int, int>> singularPoints = PoincareIndexMethod.FindSingularPoins(orientationField);
            ModelBasedAlgorithm modelBasedAlgorithm = new ModelBasedAlgorithm(orientationField);

            singularPoints = modelBasedAlgorithm.FindSingularPoints(singularPoints);
        }
    }
}
