using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Numerics;
using System.Diagnostics;
using System.Data.SqlTypes;
using System.Linq;

namespace ComplexFilterQA
{
    public static class ImageEnhancementHelper
    {
        private static double sigma1 = 0.6;
        private static double sigma2 = 3.2;
        private static double K = 1.3;
        static double tau1 = 0.1;
        static double tau2 = 0.3;
        static int ringInnerRadius = 4;
        static int ringOuterRadius = 6;

        static double sigmaDirection = 2d;
        static int directionSize = KernelHelper.GetKernelSizeForGaussianSigma(sigmaDirection);

        private static Point[,] directions = new Point[directionSize, 20];

        public static double[,] EnhanceImage(double[,] imgBytes)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            var g1 = Reduce2(imgBytes, 1.7d);
            var g2 = Reduce2(g1, 1.21d);
            var g3 = Reduce2(g2, K);
            var g4 = Reduce2(g3, K);

            var p3 = Expand2(g4, K, new Size(g3.GetLength(0), g3.GetLength(1)));
            var p2 = Expand2(g3, K, new Size(g2.GetLength(0), g2.GetLength(1)));
            var p1 = Expand2(g2, 1.21d, new Size(g1.GetLength(0), g1.GetLength(1)));

            var l3 = ContrastEnhancement(KernelHelper.Subtract(g3, p3));
            var l2 = ContrastEnhancement(KernelHelper.Subtract(g2, p2));
            var l1 = ContrastEnhancement(KernelHelper.Subtract(g1, p1));
            //SaveArray(l3, "C:\\temp\\l3.png");
            //SaveArray(l1, "C:\\temp\\l1.png");
            //SaveArray(l2, "C:\\temp\\l2.png");

            var ls1 = Symmetry.EstimateLS(l1, sigma1, sigma2);
            var ls2 = Symmetry.EstimateLS(l2, sigma1, sigma2);
            var ls3 = Symmetry.EstimateLS(l3, sigma1, sigma2);

            //SaveComplexArrayAsHSV(ls1, "C:\\temp\\ls1.png");
            //SaveComplexArrayAsHSV(ls2, "C:\\temp\\ls2.png");
            //SaveComplexArrayAsHSV(ls3, "C:\\temp\\ls3.png");

            var ls2Scaled =
                KernelHelper.MakeComplexFromDouble(
                    Expand2(ls2.Select2D(x => x.Real), K, new Size(l1.GetLength(0), l1.GetLength(1))),
                    Expand2(ls2.Select2D(x => x.Imaginary), K, new Size(l1.GetLength(0), l1.GetLength(1))));
            var multiplier = KernelHelper.Subtract(ls1.Select2D(x => x.Phase), ls2Scaled.Select2D(x => x.Phase));

            for (int x = 0; x < ls1.GetLength(0); x++)
            {
                for (int y = 0; y < ls1.GetLength(1); y++)
                {
                    ls1[x, y] *= Math.Abs(Math.Cos(multiplier[x, y]));
                }
            }

            DirectionFiltering(l1, ls1, tau1, tau2);
            DirectionFiltering(l2, ls2, tau1, tau2);
            DirectionFiltering(l3, ls3, tau1, tau2);

            var ll2 = Expand2(l3, K, new Size(l2.GetLength(0), l2.GetLength(1)));
            l2 = KernelHelper.Add(ll2, l2);
            var ll1 = Expand2(l2, 1.21d, new Size(l1.GetLength(0), l1.GetLength(1)));
            l1 = KernelHelper.Add(ll1, l1);
            var ll0 = Expand2(l1, 1.7d, new Size(imgBytes.GetLength(0), imgBytes.GetLength(1)));

            ll0 = ContrastEnhancement(ll0);
            sw.Stop();
            var enhanced = RearrangeArray(ll0, 0, 255);
            return enhanced;
        }

        private static void DirectionFiltering(double[,] l1, Complex[,] ls, double tau1, double tau2)
        {
            var l1Copy = new double[l1.GetLength(0), l1.GetLength(1)];
            for (int x = 0; x < l1.GetLength(0); x++)
            {
                for (int y = 0; y < l1.GetLength(1); y++)
                {
                    l1Copy[x, y] = l1[x, y];
                }
            }

            var kernel = new double[directionSize];
            var ksum = 0d;
            for (int i = 0; i < directionSize; i++)
            {
                ksum += kernel[i] = Gaussian.Gaussian1D(i - directionSize / 2, sigmaDirection);
            }

            for (int i = 0; i < directionSize; i++)
            {
                kernel[i] /= ksum;
            }
            for (int x = 0; x < l1.GetLength(0); x++)
            {
                for (int y = 0; y < l1.GetLength(1); y++)
                {
                    if (ls[x, y].Magnitude < tau1)
                    {
                        l1[x, y] = 0;
                    }
                    else
                    {
                        double sum = 0;
                        int area = 0;
                        for (int dx = -ringOuterRadius; dx <= ringOuterRadius; dx++)
                        {
                            for (int dy = -ringOuterRadius; dy <= ringOuterRadius; dy++)
                            {
                                if (Math.Abs(dy) < ringInnerRadius || Math.Abs(dx) < ringInnerRadius) continue;
                                int xx = x + dx;
                                if (xx < 0) xx = 0;
                                if (xx >= l1.GetLength(0)) xx = l1.GetLength(0) - 1;
                                int yy = y + dy;
                                if (yy < 0) yy = 0;
                                if (yy >= l1.GetLength(1)) yy = l1.GetLength(1) - 1;
                                sum += ls[xx, yy].Magnitude;
                                area++;
                            }
                        }
                        if (sum / area < tau2) l1[x, y] = 0;
                        else
                        {
                            var phase = ls[x, y].Phase / 2 - Math.PI / 2;
                            if (phase > Math.PI * 39 / 40) phase -= Math.PI;
                            if (phase < -Math.PI / 40) phase += Math.PI;
                            var direction = (int)Math.Round(phase / (Math.PI / 20));

                            var avg = 0.0d;
                            for (int i = 0; i < directionSize; i++)
                            {
                                var p = directions[i, direction];
                                int xx = x + p.X;
                                if (xx < 0) xx = 0;
                                if (xx >= l1.GetLength(0)) xx = l1.GetLength(0) - 1;
                                int yy = y - p.Y;
                                if (yy < 0) yy = 0;
                                if (yy >= l1.GetLength(1)) yy = l1.GetLength(1) - 1;
                                avg += kernel[i] * l1Copy[xx, yy];
                            }
                            l1[x, y] = avg;
                        }
                    }
                }
            }
        }

        private static double[,] ContrastEnhancement(double[,] source)
        {
            return source.Select2D(x => Math.Sign(x) * Math.Sqrt(Math.Abs(x)));
        }

        public static double[,] Reduce2(double[,] source, double factor)
        {

            var smoothed = ConvolutionHelper.Convolve(source,
                                                      KernelHelper.MakeKernel(
                                                          (x, y) => Gaussian.Gaussian2D(x, y, factor / 2d * 0.75d), KernelHelper.GetKernelSizeForGaussianSigma(factor / 2d * 0.75d)));
            var result = new double[(int)(source.GetLength(0) / factor), (int)(source.GetLength(1) / factor)];
            Resize(smoothed, result, factor, (x, y) => Gaussian.Gaussian2D(x, y, factor / 2d * 0.75d));
            return result;
        }

        private static double[,] Expand2(double[,] source, double factor, Size requestedSize = default(Size))
        {
            double[,] result = requestedSize == default(Size)
                                   ? new double[(int)(source.GetLength(0) * factor), (int)(source.GetLength(1) * factor)]
                                   : new double[requestedSize.Width, requestedSize.Height];
            Resize(source, result, 1 / factor, (x, y) => Gaussian.Gaussian2D(x, y, factor / 2d * 0.75d));
            return result;
        }

        public static double[,] RearrangeArray(double[,] data, double min, double max)
        {
            var dataMax = double.NegativeInfinity;
            var dataMin = double.PositiveInfinity;
            foreach (var num in data)
            {
                if (num > dataMax) dataMax = num;
                if (num < dataMin) dataMin = num;
            }
            return data.Select2D((value, row, column) => ((value - dataMin) / (dataMax - dataMin) * (max - min)) + min);
        }

        private static void Resize(double[,] source, double[,] result, double cellSize, Func<double, double, double> filterFunction)
        {
            for (int row = 0; row < result.GetLength(0); row++)
            {
                for (int column = 0; column < result.GetLength(1); column++)
                {
                    double x = cellSize * row;
                    double y = cellSize * column;

                    double sum = 0;
                    double filterSum = 0;

                    for (int xm = (int)x - 5; xm <= (int)x + 5; xm++)
                    {
                        if (xm < 0) continue;
                        if (xm >= source.GetLength(0)) break;
                        for (int ym = (int)y - 5; ym <= (int)y + 5; ym++)
                        {
                            if (ym < 0) continue;
                            if (ym >= source.GetLength(1)) break;
                            var filterValue = filterFunction(x - xm, y - ym);
                            filterSum += filterValue;
                            sum += source[xm, ym] * filterValue;
                        }
                    }
                    sum /= filterSum;
                    result[row, column] = sum;
                }
            }
        }
    }
}
