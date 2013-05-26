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

            for (int i = 0; i < 1 /*pathes.GetLength(0)*/; i++)
            {
                Tuple<int, int> redPoint = ImageHelper.FindRedPoint(pathes[i]);
                double[,] imgBytes = ImageEnhancementHelper.EnhanceImage(ImageHelper.LoadImage(pathes[i]));
                double[,] orientationField = GenerateOrientationField(imgBytes);
                List<Tuple<int, int>> singularPoints = PoincareIndexMethod.FindSingularPoins(orientationField);
                ModelBasedAlgorithm modelBasedAlgorithm = new ModelBasedAlgorithm(orientationField);
                //singularPoints = modelBasedAlgorithm.FindSingularPoints(singularPoints);

                Tuple<int, int> corePoint = modelBasedAlgorithm.FindSingularPoints(singularPoints);
                writer.WriteLine(GetDistance(redPoint, corePoint));

                //ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\china\\orientationField.jpg");
            }

            writer.Close();
        }

        private static double[,] GenerateOrientationField(double[,] bytes)
        {
            double size = 1;
            double avSigma = 5;
            double value = 1 / ((double)Constants.N * Constants.N);
            var array = new double[Constants.N, Constants.N];

            array = array.Select2D(x => value);

            var kernelAv = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, avSigma),
                                                   KernelHelper.GetKernelSizeForGaussianSigma(avSigma));

            var kernelX = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, size) * x,
                                                  KernelHelper.GetKernelSizeForGaussianSigma(size));

            var dx = ConvolutionHelper.Convolve(bytes, kernelX);

            var kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, size) * -y,
                                                  KernelHelper.GetKernelSizeForGaussianSigma(size));

            var dy = ConvolutionHelper.Convolve(bytes, kernelY);

            var Gxx = dx.Select2D(x => x * x);

            var Gxy = dx.Select2D((x, row, column) => x * dy[row, column]);

            var Gyy = dy.Select2D(x => x * x);

            Gxx = ConvolutionHelper.Convolve(Gxx, array);
            Gxy = ConvolutionHelper.Convolve(Gxy, array);
            Gyy = ConvolutionHelper.Convolve(Gyy, array);

            var angles = Gxx.Select2D((gxx, row, column) => 0.5 * Math.Atan2(gxx - Gyy[row, column], 2.0 * Gxy[row, column]));
            angles = angles.Select2D(angle => angle <= 0 ? angle + Math.PI / 2 : angle - Math.PI / 2);
            //ImageHelper.SaveArray(angles, "C:\\temp\\angles.png");
            return angles;
        }

        private static double GetDistance(Tuple<int, int> a, Tuple<int, int> b)
        {
            return Math.Sqrt(Math.Pow((a.Item1 - b.Item1), 2) + Math.Pow((a.Item2 - b.Item2), 2));
        }
    }
}
