using System;
using System.Collections.Generic;
using System.IO;
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
            string[] pathes = Directory.GetFiles("C:\\Users\\Tanya\\Documents\\tests_data\\db");
            StreamWriter writer = new StreamWriter("C:\\Users\\Tanya\\Documents\\Results\\ModelBasedAlgorithmResult.txt", true);

            for (int i = 0; i < 10 /*pathes.GetLength(0)*/; i++)
            {
                Tuple<int, int> redPoint = ImageHelper.FindRedPoint(pathes[i]);
                double[,] imgBytes = ImageEnhancementHelper.EnhanceImage(ImageHelper.LoadImage(pathes[i]));
                double[,] orientationField = PixelwiseOrientationFieldGenerator.GenerateOrientationField(imgBytes);
                List<Tuple<int, int>> singularPoints = PoincareIndexMethod.FindSingularPoins(orientationField);
                ModelBasedAlgorithm modelBasedAlgorithm = new ModelBasedAlgorithm(orientationField);
                List<Tuple<int, int>> corePoints = modelBasedAlgorithm.FindSingularPoints(singularPoints);

                writer.WriteLine("---Fingerprint---");

                foreach (Tuple<int, int> corePoint in corePoints)
                {
                    writer.WriteLine(GetDistance(redPoint, corePoint));
                }

                //ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\china\\orientationField.jpg");
            }

            writer.Close();
        }

        private static double GetDistance(Tuple<int, int> a, Tuple<int, int> b)
        {
            return Math.Sqrt(Math.Pow((a.Item1 - b.Item1), 2) + Math.Pow((a.Item2 - b.Item2), 2));
        }
    }
}
