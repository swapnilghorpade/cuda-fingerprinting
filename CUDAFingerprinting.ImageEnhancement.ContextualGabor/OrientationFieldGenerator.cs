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
        private const int W = 17;
        // Sigma of Gaussian's blur
        // from 0.34 to 0.65 for recommended size of low-pass filter
        private const double sigma = 0.65; 



        public static double[,] GenerateLeastSquareEstimate(int[,] image)
        {
            int maxY = image.GetLength(0);
            int maxX = image.GetLength(1);
            var result = new double[maxY / W, maxX / W];
            var gradX = GradientHelper.GenerateXGradient(image);
            var gradY = GradientHelper.GenerateYGradient(image);

            for (int i = 0; i < maxY / W; i++)
            {
                for (int j = 0; j < maxX / W; j++)
                {
                    double vx = 0;
                    double vy = 0;

                    for (int dy = -W / 2; dy < W / 2; dy++)
                    {
                        for (int dx = -W / 2; dx <= W / 2; dx++)
                        {
                            int y = i * W + dy;
                            int x = j * W + dx;

                            if (!(x < 0 || y < 0 || x >= maxX || y >= maxY))
                            {
                                vx += 2 * gradX[y, x] * gradY[y, x];
                                vy += Math.Pow(gradX[y, x], 2) - Math.Pow(gradY[y, x], 2);
                            }
                        }
                    }
                   // result[i, j] = (0.5 * Math.Atan2(vy, vx) >= 0) ? 0.5 * Math.Atan2(vy, vx) : 0.5 * Math.Atan2(vy, vx) + 2 * Math.PI;
                    result[i, j] = 0.5 * Math.Atan2(vy, vx);
                }
            }

            return result;
        }

        private static double[,] GenerateGaussianKernel(double sigma)
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
                            double value = 0;
                            if (x < 0 || x >= maxX || y < 0 || y >= maxY)
                            {
                                if ((x < 0 && y < 0) || (x >= maxX && y >= maxY) || (x >= maxX && y < 0) || (x < 0 && y >= maxY))
                                    value = smth[y - 2 * dy, x - 2 * dx];
                                else if (x < 0 || x >= maxX)
                                    value = smth[y, x - 2 * dx];
                                else
                                    value = smth[y - 2 * dy, x];
                            }
                            else
                                value = smth[y, x];
                            result[i, j] += value * gKernel[dy + size, dx + size];
                        }
                    }
                }
            }
            return result;
        }

        // Blur with some fixed step (use for frame's blur)
        public static double[,] GenerateBlur(double[,] smth, int step)
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
                            int x = j + dx * step;
                            int y = i + dy * step;
                            double value = 0;
                            if (x < 0 || x >= maxX || y < 0 || y >= maxY)
                            {
                                if ((x < 0 && y < 0) || (x >= maxX && y >= maxY) || (x >= maxX && y < 0) || (x < 0 && y >= maxY))
                                    value = smth[y - 2 * dy * step, x - 2 * dx * step];
                                else if (x < 0 || x >= maxX)
                                    value = smth[y, x - 2 * dx * step];
                                else
                                    value = smth[y - 2 * dy * step, x];
                            }
                            else
                                value = smth[y, x];
                            result[i, j] += value * gKernel[dy + size, dx + size];
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
            double[,] lro = new double[maxX, maxY];
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
