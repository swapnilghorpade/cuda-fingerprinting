using System;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinning
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
                        theta[i, j] = Math.Atan(newY/newX)*(180/3.14159265359); //Math.PI);
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

        public static double[,] Inv(double [,] img)
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

/*
        //local bin
        public static double[,] DoLocalBinarization(double[,] img, double[,] boardImg, int sizeWin)
        {
            double[,] mainResImg = new double[img.GetLength(0),img.GetLength(1)];
            for (int i = 0; i < mainResImg.GetLength(0); i++)
            {
                for (int j = 0; j < mainResImg.GetLength(1); j++)
                {
                    mainResImg[i, j] = 255;
                }
            }

            int itr = 0;


            bool[,] boolMask = new bool[img.GetLength(0),img.GetLength(1)];

            double[,] newImg = new double[img.GetLength(0),img.GetLength(1)];

            //1 находим самую тёмную точку
            int darkestX = 0;
            int darkestY = 0;
            for (int i = 0; i < img.GetLength(0); i++)
            {
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    if (img[i, j] < img[darkestX, darkestY])
                    {
                        darkestX = i;
                        darkestY = j;
                    }
                }
            }

            //darkestX = img.GetLength(0)/4;
            //darkestY = img.GetLength(1)/4;

            while (true)
            {
                //2 задаём окно (возможен выход за границу)
                double[,] win = new double[sizeWin,sizeWin];
                double[,] boardWin = new double[sizeWin,sizeWin];

                for (int i = 0; i < sizeWin; i++)
                {
                    for (int j = 0; j < sizeWin; j++)
                    {
                        if (i - sizeWin/2 + darkestX < 0 || i - sizeWin/2 + darkestX > img.GetLength(0) ||
                            j - sizeWin/2 + darkestY < 0 || j - sizeWin/2 + darkestY > img.GetLength(1))
                        {
                            win[i, j] = 255;
                            boardWin[i, j] = 255;
                        }
                        else
                        {

                            win[i, j] = img[i - sizeWin / 2 + darkestX, j - sizeWin / 2 + darkestY];
                            boardWin[i, j] = boardImg[i - sizeWin / 2 + darkestX, j - sizeWin / 2 + darkestY];
                        }
                    }
                }

                // return win;

                //3 ищем мин и макс

                int minX = 0;
                int mixY = 0;
                int maxX = 0;
                int maxY = 0;

                for (int i = 0; i < sizeWin; i++)
                {
                    if (win[0, i] < win[minX, mixY])
                    {
                        minX = 0;
                        mixY = i;
                    }
                    else if (win[0, i] > win[maxX, maxY])
                    {
                        maxX = 0;
                        maxY = i;
                    }

                    if (win[i, 0] < win[minX, mixY])
                    {
                        minX = i;
                        mixY = 0;
                    }
                    else if (win[i, 0] > win[maxX, maxY])
                    {
                        maxX = i;
                        maxY = 0;
                    }

                    if (win[sizeWin - 1, i] < win[minX, mixY])
                    {
                        minX = sizeWin - 1;
                        mixY = i;
                    }
                    else if (win[sizeWin - 1, i] > win[maxX, maxY])
                    {
                        maxX = sizeWin - 1;
                        maxY = i;
                    }

                    if (win[i, sizeWin - 1] < win[minX, mixY])
                    {
                        minX = i;
                        mixY = sizeWin - 1;
                    }
                    else if (win[i, sizeWin - 1] > win[maxX, maxY])
                    {
                        maxX = i;
                        maxY = sizeWin - 1;
                    }
                }

                //проверяем на разницу в 153

                //4
                int darkestBoardX = -1;
                int darkestBoardY = -1;
                double maxDarkWinBoard;
                for (int i = 0; i < boardWin.GetLength(0); i++)
                {
                    for (int j = 0; j < boardWin.GetLength(1); j++)
                    {
                        if (boardWin[i, j] < 1)
                        {
                            if (darkestBoardX < 0)
                            {
                                darkestBoardX = i;
                                darkestBoardY = j;
                            }
                            else
                            {
                                if (win[i, j] < win[darkestBoardX, darkestBoardY])
                                {
                                    darkestBoardX = i;
                                    darkestBoardY = j;
                                }
                            }
                        }
                    }
                }

                if (darkestBoardX < 0)
                {
                    maxDarkWinBoard = 0;
                }
                else
                {
                    maxDarkWinBoard = win[darkestBoardX, darkestBoardY];
                }

                double[,] copyWin = new double[win.GetLength(0),win.GetLength(1)];

                for (int i = 0; i < win.GetLength(0); i++)
                {
                    for (int j = 0; j < win.GetLength(1); j++)
                    {

                        if (win[i, j] > 153)
                        {
                            copyWin[i, j] = 255;
                        }
                        else
                        {
                            copyWin[i, j] = win[i, j];
                        }
                    }
                }

                double[,] copyWin2 = new double[win.GetLength(0),win.GetLength(1)];

                for (int i = 0; i < win.GetLength(0); i++)
                {
                    for (int j = 0; j < win.GetLength(1); j++)
                    {
                        if (win[i, j] < maxDarkWinBoard)
                        {
                            copyWin2[i, j] = win[i, j];
                        }
                        else
                        {
                            copyWin2[i, j] = 255;
                        }
                    }
                }

                //7
                double[,] resImg = new double[win.GetLength(0),win.GetLength(1)];

                for (int i = 0; i < win.GetLength(0); i++)
                {
                    for (int j = 0; j < win.GetLength(1); j++)
                    {
                        if (copyWin[i, j] < 255 || copyWin2[i, j] < 255)
                        {
                            resImg[i, j] = 0; // win[i, j];
                        }
                        else
                        {
                            resImg[i, j] = 255;
                        }
                    }
                }

                //8 результат засовываем в общее изображение
                bool isEnd = true;
                for (int i = 0; i < resImg.GetLength(0); i++)
                {
                    for (int j = 0; j < resImg.GetLength(1); j++)
                    {
                        if (i - sizeWin/2 + darkestX > 0 && i - sizeWin/2 + darkestX < mainResImg.GetLength(0) &&
                            j - sizeWin/2 + darkestY > 0 && j - sizeWin/2 + darkestY < mainResImg.GetLength(1))
                        {
                            if (!boolMask[i - sizeWin / 2 + darkestX, j - sizeWin / 2 + darkestY])
                            {
                                boolMask[i - sizeWin / 2 + darkestX, j - sizeWin / 2 + darkestY] = true;
                                mainResImg[i - sizeWin/2 + darkestX, j - sizeWin/2 + darkestY] = resImg[i, j];
                                isEnd = false;
                            }
                        }
                    }
                }
                itr++;
            }

        return mainResImg;

            //return newImg;
        }*/
   
        public static double[,] LocalBinarization(double[,] img, double[,] borderImg, int sizeWin, double scaleM)
        {
            int numberSubImgX = img.GetLength(0)/sizeWin + (img.GetLength(0)%sizeWin > 0 ? 1 : 0);
            int numberSubImgY = img.GetLength(1)/sizeWin + (img.GetLength(1)%sizeWin > 0 ? 1 : 0);

            double [,] resImg = new double[img.GetLength(0), img.GetLength(1)];

            //делаем проход по окнам (окна в нахлёст с коэф. scalrM)
            for (int i = 0; i < numberSubImgX; i++)
            {
                for (int j = 0; j < numberSubImgY; j++)
                {
                    //получаем окно greyScale
                    double [,] curWinImg = copyWin(img, i*sizeWin, j*sizeWin,  Convert.ToInt32(sizeWin*scaleM), Convert.ToInt32(sizeWin*scaleM));
                    //получаем окно с границами
                    double[,] curborderWinImg = copyWin(borderImg, i * sizeWin, j * sizeWin, Convert.ToInt32(sizeWin * scaleM), Convert.ToInt32(sizeWin * scaleM));


                    //ищем мин и макс на границах изображения ?

                    // ищем самый тёмный пиксель на границах
                    double darkestColor = FindDarkestColorBorder(curWinImg, curborderWinImg);

                    //копируем текущее окно и осветляем всё что > 153
                    double[,] clarificationImg = ClarificationImg(curWinImg, 153);

                    //копируем текущее окно и заливаем там всё что меньше darkestColor
                    double[,] fillImg = FillingImg(curWinImg, darkestColor);

                    //объединяем
                    double[,] resWin = CombinBinImgs(fillImg, clarificationImg);

                    //добавляем текущее окно в результир. изобр.
                    AddResWinTiImg(resImg, resWin, i*sizeWin, j*sizeWin, sizeWin, sizeWin);
                }
            }

            return resImg;
        }

        //копирование окна
        public static double[,] copyWin(double[,] fromImg, int xStart, int yStart, int numPixX, int numPixY)
        {
            double[,] resImg = new double[numPixX, numPixY];

            for (int i = 0; i < numPixX; i++)
            {
                for (int j = 0; j < numPixY; j++)
                {
                    resImg[i, j] = (xStart + i < 0 || xStart + i > (fromImg.GetLength(0) - 1) ||
                        yStart + j < 0 || yStart + j > (fromImg.GetLength(1) - 1)) ?  255 : fromImg[xStart + i, yStart + j];
                }    
            }

            return resImg;
        }
    
        public static double FindDarkestColorBorder(double[,] greyScaleImg, double[,] borderImg)
        {
            double darkestColor = -1;
            for (int i = 0; i < borderImg.GetLength(0); i++)
            {
                for (int j = 0; j < borderImg.GetLength(1); j++)
                {
                    if (borderImg[i, j] < 1)
                    {
                        darkestColor = darkestColor < 0 ? greyScaleImg[i, j] : darkestColor > greyScaleImg[i, j] ? greyScaleImg[i, j] : darkestColor;
                    }
                }
            }
            return darkestColor;
        }

        public static double [,] ClarificationImg(double [,] img, double border)
        {
            double[,] clarificationImg = copyWin(img, 0, 0, img.GetLength(0), img.GetLength(1));

            for (int i = 0; i < clarificationImg.GetLength(0); i++)
            {
                for (int j = 0; j < clarificationImg.GetLength(1); j++)
                {
                    clarificationImg[i, j] = clarificationImg[i, j] > border ? 255 : 0;
                }
            }

            return clarificationImg;
        }

        public static double[,] FillingImg(double[,] img, double border)
        {
            double[,] fillImg = copyWin(img, 0, 0, img.GetLength(0), img.GetLength(1));

            for (int i = 0; i < fillImg.GetLength(0); i++)
            {
                for (int j = 0; j < fillImg.GetLength(1); j++)
                {
                    fillImg[i, j] = fillImg[i, j] < border ? 0 : 255;
                }
            }
            return fillImg;
        }

        public static double[,] CombinBinImgs(double[,] img1, double[,] img2)
        {
            double [,] resImg = new double[img1.GetLength(0),img1.GetLength(1)];
            
            for (int i = 0; i < img1.GetLength(0); i++)
            {
                for (int j = 0; j < img1.GetLength(1); j++)
                {
                    resImg[i, j] = (img1[i, j] < 1 || img2[i,j] < 1) ? 0 : 255;
                }
            }

            return resImg;
        }

        public static void AddResWinTiImg(double[,] resImg, double[,] fromImg, int xStart, int yStart, int numPixX,
                                               int numPixY)
        {
            for (int i = 0; i < numPixX; i++)
            {
                for (int j = 0; j < numPixY; j++)
                {
                    if ((xStart + i < 0 || xStart + i > (resImg.GetLength(0) - 1) ||
                         yStart + j < 0 || yStart + j > (resImg.GetLength(1) - 1)))
                    {
                        break;
                    }
                    resImg[xStart + i,  yStart + j] = fromImg[i,j];
                }
            }
        }
    }
}
