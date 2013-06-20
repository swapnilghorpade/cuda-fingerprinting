using System;
using System.Collections.Generic;
using System.IO;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.ImageEnhancement.LinearSymmetry;

namespace ModelBasedAlgorithmCUDAFingerprinting.ReferencePoint.HoughPoincare
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] pathes = Directory.GetFiles("C:\\Users\\Tanya\\Documents\\tests_data\\db");
            StreamWriter writer = new StreamWriter("C:\\Users\\Tanya\\Documents\\Results\\ModelBasedAlgorithmResult.txt", true);
            StreamWriter writerResult = new StreamWriter("C:\\Users\\Tanya\\Documents\\Results\\ModelBasedAlgorithmSummaryResult.txt", true);
            List<double> distances = new List<double>();
            double sum15 = 0;
            double sum30 = 0;
            double sumOther = 0;
            double distance = 0;

            for (int i = 0; i < pathes.GetLength(0); i++)
            {
                break;
                Tuple<int, int> redPoint = ImageHelper.FindRedPoint(pathes[i]);
                double[,] imgBytes = ImageEnhancementHelper.EnhanceImage(ImageHelper.LoadImage(pathes[i]));
                double[,] orientationField = PixelwiseOrientationFieldGenerator.GenerateOrientationField(imgBytes);
                List<Tuple<int, int>> singularPoints = PoincareIndexMethod.FindSingularPoins(orientationField);
                ModelBasedAlgorithmCUDAFingerprinting.ReferencePoint.HoughPoincare.ModelBasedAlgorithm modelBasedAlgorithm = new ModelBasedAlgorithmCUDAFingerprinting.ReferencePoint.HoughPoincare.ModelBasedAlgorithm(orientationField);
                List<Tuple<int, int>> corePoints = modelBasedAlgorithm.FindSingularPoints(singularPoints);

                writer.WriteLine("---Fingerprint---");

                foreach (Tuple<int, int> corePoint in corePoints)
                {
                    distance = GetDistance(redPoint, corePoint);
                    distances.Add(distance);
                    writer.WriteLine(distance);
                }

                //ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\china\\orientationField.jpg");
            }


            foreach (double d in distances)
            {
                if (d <= 15)
                {
                    sum15 += d;
                }
                else if (d <= 30)
                {
                    sum30 += d;
                }
                else
                {
                    sumOther += d;
                }
            }

            writerResult.WriteLine("0..15 => {0}", sum15);
            writerResult.WriteLine("15..30 => {0}", sum30);
            writerResult.WriteLine(">30 => {0}", sumOther);

            writer.Close();
        }

        private static double GetDistance(Tuple<int, int> a, Tuple<int, int> b)
        {
            return Math.Sqrt(Math.Pow((a.Item1 - b.Item1), 2) + Math.Pow((a.Item2 - b.Item2), 2));
        }
    }
}
