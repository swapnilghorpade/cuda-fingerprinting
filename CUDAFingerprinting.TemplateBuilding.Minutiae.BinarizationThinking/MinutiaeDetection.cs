using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public static class MinutiaeDetection
    {
        public static int CheckMinutiae(double[,] area)
        {
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
                    newPicture[i, j] = picture[i - 1, j - 1];
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
                                area[k, l] = newPicture[i - 1 + k, j - 1 + l];
                            }
                        }
                        if (CheckMinutiae(area) > 0)
                        {
                            Minutia newMinutiae = new Minutia();
                            newMinutiae.X = i;
                            newMinutiae.Y = j;
                            minutiae.Add(newMinutiae);
                        }
                    }
                }
            }
            return minutiae;
        }
    }
}
