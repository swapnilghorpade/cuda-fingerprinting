using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public static class Thining
    {
        public static double B(double[,] picture, int x, int y)        //Метод В(Р) возвращает количество черных пикселей в окрестности точки Р
        {
            return picture[x, y - 1] + picture[x + 1, y - 1] + picture[x + 1, y] + picture[x + 1, y + 1] +
                   picture[x, y + 1] + picture[x - 1, y + 1] + picture[x - 1, y] + picture[x - 1, y - 1];
        }

        public static double A(double[,] picture, int x, int y)        //Метод А(Р) возвращает количество подряд идущих белых и черных пикселей вокруг точки Р (..0->1..)
        {
            int counter = 0;
            if((picture[x, y - 1] == 0) && (picture[x + 1, y - 1] == 1))
            {
                counter++;
            }
            if ((picture[x + 1, y - 1] == 0) && (picture[x + 1, y] == 1))
            {
                counter++;
            }
            if ((picture[x + 1, y] == 0) && (picture[x + 1, y + 1] == 1))
            {
                counter++;
            }
            if ((picture[x + 1, y + 1] == 0) && (picture[x, y + 1] == 1))
            {
                counter++;
            }
            if ((picture[x, y + 1] == 0) && (picture[x - 1, y + 1] == 1))
            {
                counter++;
            }
            if ((picture[x - 1, y + 1] == 0) && (picture[x - 1, y] == 1))
            {
                counter++;
            }
            if ((picture[x - 1, y] == 0) && (picture[x - 1, y - 1] == 1))
            {
                counter++;
            }
            if ((picture[x - 1, y - 1] == 0) && (picture[x, y - 1] == 1))
            {
                counter++;
            }
            return counter;
        }

        public static double[,] ThiningPicture(double[,] newPicture)
        {
            
            double[,] picture = new double[newPicture.GetLength(0) + 2,newPicture.GetLength(1) + 2];
            double[,] pictureToRemove = new double[newPicture.GetLength(0) + 2, newPicture.GetLength(1) + 2];
            bool hasChanged;
            for (int i = 0; i < picture.GetLength(1); i++)
            {
                for (int j = 0; j < picture.GetLength(0); j++)
                {
                    picture[j, i] = 255;
                    pictureToRemove[j, i] = 0;
                }
            }

            for (int i = 0; i < newPicture.GetLength(1); i++)
            {
                for (int j = 0; j < newPicture.GetLength(0); j++)      //Вставляем входную картинку в более широкий массив для удобства рассмотрения граничных случаев 
                {
                    picture[j + 1, i + 1] = newPicture[j, i];
                }
            }
            
            for (int i = 0; i < picture.GetLength(1); i++)
            {
                for (int j = 0; j < picture.GetLength(0); j++)
                {
                    picture[j, i] = picture[j, i] == 0 ? picture[j, i] = 1 : picture[j, i] = 0;   //Черным клеткам присваиваем значения 1, белым - 0;
                }
            }
            do
            {
                hasChanged = false;
                for (int i = 0; i < newPicture.GetLength(1); i++)
                {
                    for (int j = 0; j < newPicture.GetLength(0); j++)
                    {
                        if ((picture[j, i] == 1) && (2 <= B(picture, j, i)) && (B(picture, j, i) <= 6) && (A(picture, j, i) == 1) &&                          //Непосредственное удаление точки, см. Zhang-Suen thinning algorithm, http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf
                            (picture[j, i - 1] * picture[j + 1, i] * picture[j, i + 1] == 0) &&
                            (picture[j + 1, i] * picture[j, i + 1] * picture[j - 1, i] == 0))
                        {
                            pictureToRemove[j, i] = 1;
                            hasChanged = true;
                        }
                    }
                }
                for (int i = 0; i < newPicture.GetLength(1); i++)
                {
                    for (int j = 0; j < newPicture.GetLength(0); j++)
                    {
                        if (pictureToRemove[j, i] == 1)
                        {
                            picture[j, i] = 0;
                            pictureToRemove[j, i] = 0;
                        }
                    }
                }
                for (int i = 0; i < newPicture.GetLength(1); i++)
                {
                    for (int j = 0; j < newPicture.GetLength(0); j++)
                    {
                        if ((picture[j, i] == 1) && (2 <= B(picture, j, i)) && (B(picture, j, i) <= 6) &&
                            (A(picture, j, i) == 1) &&
                            (picture[j, i - 1] * picture[j + 1, i] * picture[j - 1, i] == 0) &&
                            (picture[j, i - 1] * picture[j, i + 1] * picture[j - 1, i] == 0))
                        {
                            pictureToRemove[j, i] = 1;
                            hasChanged = true;
                        }
                    }
                }

                for (int i = 0; i < newPicture.GetLength(1); i++)
                {
                    for (int j = 0; j < newPicture.GetLength(0); j++)
                    {
                        if (pictureToRemove[j, i] == 1)
                        {
                            picture[j, i] = 0;
                            pictureToRemove[j, i] = 0;
                        }
                    }
                }
            } while (hasChanged);

            for (int i = 0; i < newPicture.GetLength(1); i++)
            {
                for (int j = 0; j < newPicture.GetLength(0); j++)
                {
                    if ((picture[j, i] == 1) &&
                        (((picture[j, i - 1] * picture[j + 1, i] == 1) && (picture[j - 1, i + 1] != 1)) || ((picture[j + 1, i] * picture[j, i + 1] == 1) && (picture[j - 1, i - 1] != 1)) ||      //Небольшая модификцаия алгоритма для ещё большего утоньшения
                        ((picture[j, i + 1] * picture[j - 1, i] == 1) && (picture[j + 1, i - 1] != 1)) || ((picture[j, i - 1] * picture[j - 1, i] == 1) && (picture[j + 1, i + 1] != 1))))
                    {
                        picture[j, i] = 0;
                    }
                }
            }


            //for (int j = 0; j < newPicture.GetLength(0); j++)
            //{
            //    for (int i = 0; i < newPicture.GetLength(1); i++)
            //    {
            //        System.Console.Write(picture[j, i] + " ");
            //    }
            //    System.Console.WriteLine("\n");
            //}
                

            for (int i = 0; i < picture.GetLength(1); i++)
            {
                for (int j = 0; j < picture.GetLength(0); j++)
                {
                    picture[j, i] = picture[j, i] == 0 ? picture[j, i] = 255 : picture[j, i] = 0;       // Черным клеткам присваеваем значения 0, белым - 255
                }
            }
            
            double [,] outPicture = new double[newPicture.GetLength(0),newPicture.GetLength(1)];

            for (int i = 0; i < newPicture.GetLength(1); i++)
            {
                for (int j = 0; j < newPicture.GetLength(0); j++)
                {                                                                                       
                    outPicture[j, i] = picture[j + 1, i + 1];
                }
            }
            return outPicture;
        }
    }
}
