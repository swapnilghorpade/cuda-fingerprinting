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

            // double[,] pixelwiseOrientationField = PixelwiseOrientationFieldGenerator.GenerateOrientationField(imgBytes);
            double[,] orientationField = GenerateOrientationField(imgBytes, 15);

            //ImageHelper.SaveArray(pixelwiseOrientationField, "C:\\Users\\Tanya\\Documents\\Results\\china\\orientationField.jpg");
            ImageHelper.SaveArray(orientationField, "C:\\Users\\Tanya\\Documents\\Results\\china\\orientationField.jpg");

            List<Tuple<int, int>> singularPoints = PoincareIndexMethod.FindSingularPoins(orientationField);
            ModelBasedAlgorithm modelBasedAlgorithm = new ModelBasedAlgorithm(orientationField);
            singularPoints = modelBasedAlgorithm.FindSingularPoints(singularPoints);
        }

        private static double[,] GenerateOrientationField(double[,] bytes, int n)
        {
            double size = 1;
            double avSigma = 5;
            double value = 1 / ((double)n * n);
            var array = new double[n, n];

            array = array.Select2D(x => value); // GetArray(n);

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
    }
}
