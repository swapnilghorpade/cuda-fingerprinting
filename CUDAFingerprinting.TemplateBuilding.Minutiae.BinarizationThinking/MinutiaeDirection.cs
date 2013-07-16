using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking
{
    public static class MinutiaeDirection
    {
        public static void FindDirection(double[,] OrientationField, int dim, List<Minutia> Minutiae , int[,] BinaryImage)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].X;
                int curY = Minutiae[cur].Y;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))   
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count == 2) //Ending of Line
                {
                    double angle = 0;
                    for (int i = -1; i < 2; i++)
                        for (int j = -1; j < 2; j++)
                            if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                                if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                                {
                                    if (j == 0)
                                        angle = Math.PI + i*Math.PI/2;
                                    else
                                         angle = Math.Atan((double) i/j);
                                }
                    if ((Teta - angle < Math.PI/2) && (angle - Teta < Math.PI/2))
                    {
                        var temp = Minutiae[cur];
                        temp.Angle = Teta + Math.PI;
                        Minutiae[cur]=temp;
                    }
                    else
                    {
                        var temp = Minutiae[cur];
                        temp.Angle = Teta;
                        Minutiae[cur] = temp;
                    }
                }
                else //Bifurcation
                {
                    int radius = 3; //Use squares 7x7
                    int minX = 0;
                    int minY = 0;
                    double minAngle = 0;
                    double angle = 0;
                    if (Teta < Math.PI/2) //Quadrant I or III
                    {
                        for (int i = 0; i <= radius; i++) //Check Quadrant III
                            for (int j = 0; j >= (-1)*radius; j--)
                                if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                                if ((i != 0) || (j != 0))
                                    if (BinaryImage[curX + i, curY + j] == 0)
                                        if ((minX == 0) && (minY == 0))
                                        {
                                            minX = i;
                                            minY = j;
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            minAngle = Math.Abs(angle - Teta);
                                        }
                                        else
                                        {
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            if (Math.Abs(angle - Teta) < minAngle)
                                            {
                                                minAngle = Math.Abs(angle - Teta);
                                                minX = i;
                                                minY = j;
                                            }
                                        }
                        for (int i = 0; i >= (-1)*radius; i--) //Check Quadrant I
                            for (int j = 0; j <= radius; j++)
                                if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                                if ((i != 0) || (j != 0))
                                    if (BinaryImage[curX + i, curY + j] == 0)
                                        if ((minX == 0) && (minY == 0))
                                        {
                                            minX = i;
                                            minY = j;
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            minAngle = Math.Abs(angle - Teta);
                                        }
                                        else
                                        {
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            if (Math.Abs(angle - Teta) < minAngle)
                                            {
                                                minAngle = Math.Abs(angle - Teta);
                                                minX = i;
                                                minY = j;
                                            }
                                        }
                        if ((minX >= 0) && (minY <= 0))
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Teta + Math.PI;
                            Minutiae[cur] = temp;
                        }
                        else
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Teta;
                            Minutiae[cur] = temp;
                        }

                    }
                    else //Quadrant II or IV
                    {
                        for (int i = 0; i <= radius; i++) //Check Quadrant  IV
                            for (int j = 0; j <= radius; j++)
                                if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                                if ((i != 0) || (j != 0))
                                    if (BinaryImage[curX + i, curY + j] == 0)
                                        if ((minX == 0) && (minY == 0))
                                        {
                                            minX = i;
                                            minY = j;
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            minAngle = Math.Abs(angle - Teta);
                                        }
                                        else
                                        {
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            if (Math.Abs(angle - Teta) < minAngle)
                                            {
                                                minAngle = Math.Abs(angle - Teta);
                                                minX = i;
                                                minY = j;
                                            }
                                        }
                        for (int i = 0; i >= (-1)*radius; i--) //Check Quadrant  II
                            for (int j = 0; j >= (-1)*radius; j--)
                                if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                                if ((i != 0) || (j != 0))
                                    if (BinaryImage[curX + i, curY + j] == 0)
                                        if ((minX == 0) && (minY == 0))
                                        {
                                            minX = i;
                                            minY = j;
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            minAngle = Math.Abs(angle - Teta);
                                        }
                                        else
                                        {
                                            if (j == 0)
                                                angle = Math.PI + i*Math.PI/2;
                                            else
                                                angle = Math.Atan((double) i/j);
                                            if (Math.Abs(angle - Teta) < minAngle)
                                            {
                                                minAngle = Math.Abs(angle - Teta);
                                                minX = i;
                                                minY = j;
                                            }
                                        }
                        if ((minX >= 0) && (minY >= 0))
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Teta + Math.PI;
                            Minutiae[cur] = temp;
                        }
                        else
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Teta;
                            Minutiae[cur] = temp;
                        }
                    }
                }
            }
        }
    }
}
