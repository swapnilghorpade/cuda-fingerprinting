using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public static class LocalBinarizationCanny
    {
        public static double[,] theta;

        public static double[,] Smoothing(double[,] img, double sigma)
        {
            var kernel = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, sigma),
                                                   KernelHelper.GetKernelSizeForGaussianSigma(sigma));
            double[,] data = ConvolutionHelper.Convolve(img, kernel);

            return data;
        }

        public static double[,] Sobel(double[,] img)
        {

            theta = new double[img.GetLength(0), img.GetLength(1)];

            double[,] sobData = new double[img.GetLength(0), img.GetLength(1)];

            double[,] gX = new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            double[,] gY = new double[,] { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            for (int i = 1; i < img.GetLength(0) - 1; i++)
                for (int j = 1; j < img.GetLength(1) - 1; j++)
                {
                    double newX = 0, newY = 0;

                    for (int h = 0; h < 3; h++)
                    {
                        for (int w = 0; w < 3; w++)
                        {
                            double curr = img[i + h - 1, j + w - 1];
                            newX += gX[h, w] * curr;
                            newY += gY[h, w] * curr;
                        }
                    }

                    sobData[i, j] = Math.Sqrt(newX * newX + newY * newY);

                    if (newX == 0)
                    {
                        theta[i, j] = 90d;
                    }
                    else
                    {
                        theta[i, j] = Math.Atan(newY / newX);
                    }
                }
            return sobData;
        }

        public static double[,] NonMaximumSupperession(double[,] img)
        {
            double[,] newImg = new double[img.GetLength(0), img.GetLength(1)];

            for (int i = 1; i < img.GetLength(0) - 1; i++)
            {
                for (int j = 1; j < img.GetLength(1) - 1; j++)
                {
                    if (theta[i, j] > 67.5d)
                    {
                        if (theta[i, j] > 112.5d)
                        {
                            if (theta[i, j] > 157.5d)
                            {
                                theta[i, j] = 135;
                            }
                            else
                            {
                                theta[i, j] = 0;
                            }
                        }
                        else
                        {
                            theta[i, j] = 90d;
                        }
                    }
                    else
                    {
                        if (theta[i, j] > 22.5d)
                        {
                            theta[i, j] = 45d;
                        }
                        else
                        {
                            theta[i, j] = 0d;
                        }
                    }

                    int dx = Math.Sign(Math.Cos(theta[i, j]));
                    int dy = -Math.Sign(Math.Sin(theta[i, j]));

                    if (img[i, j] > img[i + dx, j + dy] && img[i, j] > img[i - dx, j - dy])
                    {
                        newImg[i, j] = img[i, j];
                    } //иначе остаётся 0
                }

            }
            return newImg;
        }


        //далее примен. глобальную бинаризацию. уже имеется

        public static double[,] Traceroute(double[,] img)
        {
            var moveDir = new int[,] { { -1, -1, -1, 0, 0, 1, 1, 1 }, { -1, 0, 1, -1, 1, -1, 0, 1 } };
            double[,] newImg = new double[img.GetLength(0), img.GetLength(1)];

            for (int i = 0; i < img.GetLength(0) - 1; i++)
            {
                for (int j = 0; j < img.GetLength(1) - 1; j++)
                {
                    if (img[i, j] > 0)
                    {
                        newImg[i, j] = img[i,j];
                        bool clear = true;
                        for (int k = 0; k < 7; k++)
                        {
                            int dx = moveDir[0, k];
                            int dy = moveDir[1, k];

                            int X = dx;
                            int Y = dy;
                            X = i + dx;
                            Y = j + dy;

                            if (X < 0 || Y < 0 || X > img.GetLength(0) - 1 || Y > img.GetLength(1) - 1)
                            {
                                continue;
                            }
                            if (img[X, Y] > 0)
                            {
                                clear = false;
                                continue;
                            }
                        }
                        if (clear)
                        {
                            newImg[i, j] = 0;
                        }
                    }
                    else
                    {
                        newImg[i, j] = img[i,j];
                    }
                }
            }
            newImg[0, 0] = 255;
            return newImg;
        }

        public static double[,] inv(double [,] img)
        {
            double [,] newImg = new double[img.GetLength(0),img.GetLength(1)];

            for (int i = 0; i < newImg.GetLength(0); i++)
            {
                for (int j = 0; j < newImg.GetLength(1); j++)
                {
                    newImg[i,j] = img[i, j] > 0 ? 0 : 255;
                }
            }
            return newImg;
        }
    }
}
