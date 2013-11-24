using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinning
{
    public static class MinutiaeDetection
    {
        public static int CheckMinutiae(double[,] area) //Данный метод определяет, является ли пиксель минуцией, result = 0 - не минуция, иначе - минуция
        {                                               //На вход дается окрестность пикселя
            int result; // 1 - ending, >2 - branching,                     
            int counter = 0;
            area[1, 1] = 255;
            for (int i = 0; i < area.GetLength(0); i++)
            {
                for (int j = 0; j < area.GetLength(1); j++)
                {
                    if (area[i, j] == 0) counter++;
                }
            }
            if (counter == 1)
            {
                return result = 1;
            }
            else
            {
                if (counter > 2)
                {
                    return result = counter;
                }
                else
                {
                    return result = 0;
                }
            }
        }
 
        public static List<Minutia> FindMinutiae(double[,] picture)
        {
            List<Minutia> minutiae = new List<Minutia>();
            double[,] area = new double[3, 3];
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    area[k, l] = 255;
                }
            }
            double[,] newPicture = new double[picture.GetLength(0) + 2, picture.GetLength(1) + 2];
            for (int i = 0; i < newPicture.GetLength(0); i++)
            {
                for (int j = 0; j < newPicture.GetLength(1); j++)
                {
                    newPicture[i, j] = 255;
                }
            }
            for (int i = 1; i < newPicture.GetLength(0) - 1; i++)
            {
                for (int j = 1; j < newPicture.GetLength(1) - 1; j++)
                {
                    newPicture[i, j] = picture[i - 1, j - 1];                              //Вставляем входную картинку в более широкий массив для удобства рассмотрения граничных случаев 
                }
            }
            for (int i = 1; i < newPicture.GetLength(0) - 1; i++)
            {
                for (int j = 1; j < newPicture.GetLength(1) - 1; j++)
                {
                    if (newPicture[i, j] == 0)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            for (int l = 0; l < 3; l++)
                            {
                                area[k, l] = newPicture[i - 1 + k, j - 1 + l];    //Проходим по массиву и проверяем для каждого черного пикселя, является ли он минуцией. 
                            }
                        }
                        if (CheckMinutiae(area) > 0)
                        {
                            Minutia newMinutiae = new Minutia();                 //Если да, то добавляем минуцию в стек
                            newMinutiae.X = j - 1;
                            newMinutiae.Y = i - 1;
                            minutiae.Add(newMinutiae);
                        }
                    }
                }
            }
            return minutiae;
        }

        public static List<Minutia> FindBigMinutiae(List<Minutia> source)
        {
            int Radius = 5;

            // this structure is used for storing indices for minutiae
            // that are closed to the minutia of the selected index
            // key of the pair is the minutia index that is the center of the circle
            var listBigMinutiae = new List<KeyValuePair<int, List<int>>>(); 

            for (int i = 0; i < source.Count; i++)
            {
                List<int> listSmallMinutiae = new List<int>();
                for (int j = 0; j < source.Count; j++)
                {
                    int dX = source[i].X - source[j].X;
                    int dY = source[i].Y - source[j].Y;
                    if ((dX * dX + dY * dY) < Radius * Radius)
                    {
                        listSmallMinutiae.Add(j);
                    }
                }
                listBigMinutiae.Add(new KeyValuePair<int, List<int>>(i, listSmallMinutiae));
            }

            listBigMinutiae = listBigMinutiae.OrderByDescending(x => x.Value.Count).ToList();

            List<Minutia> result = new List<Minutia>();
            for (int i = 0; i < listBigMinutiae.Count; i++)
            {
                if (listBigMinutiae[i].Value.Any())
                {
                    Minutia newMinutia = new Minutia();

                    var circleList = listBigMinutiae[i].Value;

                    for (int j = 0; j < listBigMinutiae[i].Value.Count; j++)
                    {
                            newMinutia.X += source[circleList[j]].X;
                            newMinutia.Y += source[circleList[j]].Y;
                    }
                    newMinutia.X /= circleList.Count;
                    newMinutia.Y /= circleList.Count;

                    for (int j = i + 1; j < listBigMinutiae.Count; j++)
                    {
                        if(circleList.Contains(listBigMinutiae[j].Key))listBigMinutiae[j].Value.Clear();
                        else
                        {
                            listBigMinutiae[j] = new KeyValuePair<int, List<int>>(listBigMinutiae[j].Key,
                                                                                  listBigMinutiae[j].Value.Except(
                                                                                      circleList).ToList());
                        }
                    }
                    result.Add(newMinutia);

                    //    dX = newMinutia.X - listBigMinutiae[i][0].X;
                    //dY = newMinutia.Y - listBigMinutiae[i][0].Y;
                    //int min = dX*dX + dY*dY;
                    //Minutia newMinutia1 = new Minutia();
                    //newMinutia1.X = listBigMinutiae[i][0].X;
                    //newMinutia1.Y = listBigMinutiae[i][0].Y;
                    //for (int j = 0; j < listBigMinutiae[i].Count; j++)
                    //{
                    //    dX = newMinutia.X - listBigMinutiae[i][j].X;
                    //    dY = newMinutia.Y - listBigMinutiae[i][j].Y;
                    //    int locmin = dX*dX + dY*dY;
                    //    if (min > locmin)
                    //    {
                    //        min = locmin;
                    //        newMinutia1.X = listBigMinutiae[i][j].X;
                    //        newMinutia1.Y = listBigMinutiae[i][j].Y;
                    //    }

                    //}
                }
            }
            return result;
        }
    }
}
