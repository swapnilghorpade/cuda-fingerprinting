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
        private static void FindDirectionOfBifurcationUsingQuadrants(int cur,double Teta,List<Minutia> Minutiae, int[,] BinaryImage)
        {
            int radius = 3; //Use squares 7x7
            int minX = 0;
            int minY = 0;
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            double minAngle = 0;
            double angle = 0;
            if (Teta < Math.PI / 2) //Quadrant I or III
            {
                for (int i = 0; i <= radius; i++) //Check Quadrant III
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                    if ((minX == 0) && (minY == 0))
                                    {
                                        minX = i;
                                        minY = j;
                                        angle = Math.Acos(((double) j)/((double) (i*i + j*j)));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
                                        if (Math.Abs(angle - Teta) < minAngle)
                                        {
                                            minAngle = Math.Abs(angle - Teta);
                                            minX = i;
                                            minY = j;
                                        }
                                    }
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant I
                    for (int j = 0; j <= radius; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                    if ((minX == 0) && (minY == 0))
                                    {
                                        minX = i;
                                        minY = j;
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
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
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
                                        if (Math.Abs(angle - Teta) < minAngle)
                                        {
                                            minAngle = Math.Abs(angle - Teta);
                                            minX = i;
                                            minY = j;
                                        }
                                    }
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant  II
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) && (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                    if ((minX == 0) && (minY == 0))
                                    {
                                        minX = i;
                                        minY = j;
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / ((double)(i * i + j * j)));
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
        
        
        
        public static void FindDirectionVersion1(double[,] OrientationField, int dim, List<Minutia> Minutiae , int[,] BinaryImage)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
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
                                    angle = Math.Asin(((double)j) / ((double)(i * i + j * j)));
                                    if (i > 0)
                                        angle += Math.PI;
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
                    FindDirectionOfBifurcationUsingQuadrants(cur,Teta,Minutiae,BinaryImage);
            }
        }
    }
}
