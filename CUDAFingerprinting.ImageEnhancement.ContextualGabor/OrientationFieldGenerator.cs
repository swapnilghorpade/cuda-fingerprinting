using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.ImageEnhancement.ContextualGabor
{
    public static class OrientationFieldGenerator
    {
        // frame's size for square estimate
        public const int W = 11;
        // Sigma of Gaussian's blur
        // from 0.34 to 0.65 for recommended size of low-pass filter
        private const double sigma = 0.7;



        public static double[,] GenerateLeastSquareEstimate(int[,] image)
        {
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            var result = new double[maxY / W, maxX / W];
            var gradX = GradientHelper.GenerateXGradient(image);
            var gradY = GradientHelper.GenerateYGradient(image);
            double[,] vx = new double[maxY / W, maxX / W];
            double[,] vy = new double[maxY / W, maxX / W];

            for (int i = 0; i < maxY / W; i++)
            {
                for (int j = 0; j < maxX / W; j++)
                {
                    vx[i, j] = 0;
                    vy[i, j] = 0;

                    for (int dy = -W / 2; dy <= W / 2; dy++)
                    {
                        for (int dx = -W / 2; dx <= W / 2; dx++)
                        {
                            // Middle of the block with specified offset
                            int y = i * W + W / 2 + dy;
                            int x = j * W + W / 2 + dx;

                            if (x >= 0 && y >= 0 && x < maxX && y < maxY)
                            {
                                vx[i, j] += 2 * gradX[y, x] * gradY[y, x];
                                vy[i, j] += -gradX[y, x] * gradX[y, x] + gradY[y, x] * gradY[y, x];
                            }
                        }
                    }

                   var hypotenuse = Math.Sqrt(vy[i, j] * vy[i, j] + vx[i, j] * vx[i, j]);
                    vx[i, j] = vx[i, j] / hypotenuse;
                    vy[i, j] = vy[i, j] / hypotenuse;

                    result[i, j] = 0.5 * Math.Atan2(vy[i, j], vx[i, j]);
                    result[i, j] = (result[i, j] < 0) ? result[i, j] + Math.PI : (result[i, j] > Math.PI ? result[i, j] - Math.PI : result[i, j]);
                }
            }

            for (int x = 0; x < maxX / W; x++)
            {
                for (int y = 0; y < maxY / W; y++)
                {
                    double resultX = 0, resultY = 0;
                    int count = 0;
                    for (int i = -1; i < 2; i++)
                    {
                        if (y + i < 0 || y + i >= maxY / W) continue;
                        for (int j = -1; j < 2; j++)
                        {
                            if (x + j < 0 || x + j >= maxX / W) continue;
                            resultX += vx[y + i, x + j];
                            resultY += vy[y + i, x + j];
                            count++;
                        }
                    }
                    var xx = resultY / count;
                    var yy = resultX / count;
                    result[y, x] = Math.Atan2(yy, xx) / 2;
                }
            }
            return result;
        }

        public static double[,] GenerateGaussianKernel(double sigma)
        {
            int size = 1 + 2 * (int)Math.Ceiling(3 * sigma);
            var result = new double[size, size];
            int center = size / 2;
            Func<int, int, double> gaussian = (x, y) => 1 / (2 * Math.PI * sigma * sigma) * Math.Exp(-(x * x + y * y) / (2 * sigma * sigma));

            for (int i = -center; i <= center; i++)
            {
                for (int j = -center; j <= center; j++)
                {
                    result[i + center, j + center] = gaussian(i, j);
                }
            }
            return result;
        }

        public static double[,] GenerateBlur(double[,] smth)
        {
            int maxY = smth.GetLength(0);
            int maxX = smth.GetLength(1);
            var result = new double[maxY, maxX];
            var gKernel = GenerateGaussianKernel(sigma); 
            
            int size = gKernel.GetLength(0) / 2;

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    for (int dy = -size; dy <= size; dy++)
                    {
                        for (int dx = -size; dx <= size; dx++)
                        {
                            int x = j + dx;
                            int y = i + dy;
                            if (x >= 0 && x < maxX && y >= 0 && y < maxY)
                            {
                                double value = smth[y, x];
                                result[i, j] += value * gKernel[dy + size, dx + size];
                            }
                        }
                    }
                }
            }
            return result;
        }

        // (x, y) pairs of continious vector field
        public static Tuple<double, double>[,] GenerateLowPassFilteredContiniousVectorField(int[,] image)
        {
            int maxY = image.GetLength(0) / W;
            int maxX = image.GetLength(1) / W;
            var cvf = new Tuple<double, double>[maxY, maxX];
            var lsq = GenerateLeastSquareEstimate(image);

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    cvf[i, j] = new Tuple<double, double>(Math.Cos(2 * lsq[i, j]), Math.Sin(2 * lsq[i, j]));
                }
            }
            return KernelHelper.Zip2D(GenerateBlur(cvf.Select2D(x => x.Item1)), GenerateBlur(cvf.Select2D(x => x.Item2)), 
                (x, y) => new Tuple<double, double>(x, y));
        }

        public static double[,] GenerateLocalRidgeOrientation(int[,] image)
        {
            int maxY = image.GetLength(0) / W;
            int maxX = image.GetLength(1) / W;
            double[,] lro = new double[maxY, maxX];
            var cvf = GenerateLowPassFilteredContiniousVectorField(image);

            for (int i = 0; i < maxY; i++)
            {
                for (int j = 0; j < maxX; j++)
                {
                    lro[i, j] = 0.5 * Math.Atan2(cvf[i, j].Item2, cvf[i, j].Item1);
                }
            }
            return lro;
        }   
    }
}
