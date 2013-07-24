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
        /*private static void FindDirectionOfBifurcationUsingQuadrants(int cur,double Teta,List<Minutia> Minutiae, int[,] BinaryImage)
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
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
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
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
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
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
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

        private static void CheckPath(int curX, int curY, int prevX, int prevY, int[,] BinaryImage,ref double score,ref
                                        int number,int radiusOfSearch)
        {
            bool StopSearch = false;
            int nextX = curX;
            int nextY = curY;
            number++;
            double angle = Math.Acos((((double)(curY-prevY))/Math.Sqrt((double)((curY-prevY)*(curY-prevY) + (curX-prevX)*(curX-prevX)))));
            if ((curX - prevX) > 0)
                angle = 2*Math.PI - angle;
            score += angle;
            if (number < radiusOfSearch)
                for (int i = -1; (i < 2) && (!StopSearch); i++)
                    for (int j = -1; (j < 2) && (!StopSearch); j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                        (curY + j < BinaryImage.GetLength(1)))
                            if (((i != 0) || (j != 0)) && ((curX + i != prevX) || (curY + j != prevY)))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                    if ((nextX == curX) && (nextY == curY))
                                    {
                                        nextX = curX + i;
                                        nextY = curY + j;
                                    }
                                    else
                                        StopSearch = true;
            if ((nextX == curX) && (nextY == curY))
                StopSearch = true;
            if (!StopSearch)
                CheckPath(nextX, nextY, curX, curY, BinaryImage,ref score,ref number, radiusOfSearch);
        }

        private static void FindDirectionOfBifurcationUsingAverageValues(int cur, double Teta, List<Minutia> Minutiae,
                                                                         int[,] BinaryImage)
        {
            int RadiusOfSearch = 7;
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            double minAngle = 2*Math.PI;
            int minI = 0;
            int minJ = 0;
            for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                    if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                        (curY + j < BinaryImage.GetLength(1)))
                        if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                        {
                            int number = 0;
                            double score = 0;
                            CheckPath(curX + i, curY + j, curX, curY, BinaryImage,ref score,ref number, RadiusOfSearch);
                            double angle = score/number;
                            if (angle > Math.PI)
                                angle = +Math.PI;
                            if (Math.Abs(angle - Teta) < minAngle)
                            {
                                minI = i;
                                minJ = j;
                                minAngle = angle;
                            }
                        }
            if ((minJ <= 0) && (minI >= 0))
            {
                var temp = Minutiae[cur];
                temp.Angle = Teta;
                Minutiae[cur] = temp;
            }
            else
            {
                var temp = Minutiae[cur];
                temp.Angle = Teta + Math.PI;
                Minutiae[cur] = temp;
            }      
        }
        

        private static void FindDirectionOfBifurcationUsingAlignment(int cur, double Teta, List<Minutia> Minutiae,
                                                                     int[,] BinaryImage,double delta,int step)
        {
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            if ((curX > 0) && (curX < BinaryImage.GetLength(0)-1) && (curY < BinaryImage.GetLength(1)-1) && (curY > 0)) {
                int[] startPointX = new int[3];
                int[] startPointY = new int[3];
                int[] curPointX = new int[3];
                int[] curPointY = new int[3];
                int[] prevPointX = new int[3];
                int[] prevPointY = new int[3];
                bool[] StopSteps = new bool[3];
                int[] Depth = new int[3];
                int num = 0;
                for (int i = -1 ; i < 2;i ++)
                    for (int j = -1 ; j < 2; j++)
                        if (((j != 0) || (i != 0)) && (BinaryImage[curX + i,curY+  j] == 0))
                        {
                            startPointX[num] = curX + i;
                            startPointY[num] = curY + j;
                            curPointX[num] = curX + i;
                            curPointY[num] = curY + j;
                            prevPointX[num] = curX;
                            prevPointY[num] = curY;
                            num++;
                        }
                StopSteps[0] = false;
                StopSteps[1] = false;
                StopSteps[2] = false;
                Depth[0] = 0;
                Depth[1] = 0;
                Depth[2] = 0;
                int number = 0;
                double score = 0;
                bool Find = false;
                while (((!StopSteps[0]) && (!StopSteps[1]) && (!StopSteps[2])) && (Depth[0] + Depth[1] + Depth[2] < 50) && (!Find))
                {
                    for (num = 0; num <3 ; num++)
                        if (!StopSteps[num])
                        {
                            score = 0;
                            number = 0;
                            for (int k = 0; (k < step) && (!StopSteps[num]); k++)
                            {
                                int nextX = 0;
                                int nextY = 0;
                                Depth[num]++;
                                number++;
                                double angle = Math.Acos((((double)(curPointY[num]-prevPointY[num]))/Math.Sqrt((double)((curPointY[num]-prevPointY[num])*(curPointY[num]-prevPointY[num]) + (curPointX[num]-prevPointX[num])*(curPointX[num]-prevPointX[num])))));
                                if ((curPointX[num] - prevPointX[num]) > 0)
                                    angle = 2*Math.PI - angle;
                                score += angle;
                                if (Math.Abs(score/num - Teta) < delta)
                                    StopSteps[num] = true;
                                for (int i = -1; (i < 2) && (!StopSteps[num]); i++)
                                    for (int j = -1; (j < 2) && (!StopSteps[num]); j++)
                                    if ((curPointX[num] + i >= 0) && (curPointX[num] + i < BinaryImage.GetLength(0)) && (curPointY[num] + j >= 0) &&
                                        (curPointY[num] + j < BinaryImage.GetLength(1)))
                                        if (((i != 0) || (j != 0)) &&
                                            (BinaryImage[curPointX[num] + i, curPointY[num] + j] == 0) &&
                                            ((curPointX[num] + i != prevPointX[num]) ||
                                         (curPointY[num] + j != prevPointY[num])))
                                        {
                                            if ((nextX == 0) && (nextY == 0))
                                            {
                                                nextX = i;
                                                nextY = j;
                                            }
                                            else
                                                StopSteps[num] = true;
                                        }
                                if ((nextX == 0) && (nextY == 0) && (!StopSteps[num]))
                                {
                                    Find = true;
                                    int p = 0;
                                    while (((Minutiae[p].X != curPointX[num]) || (Minutiae[p].Y != curPointY[num])) && (p<Minutiae.Count()-1))
                                        p++;
                                    if ((Minutiae[p].X == curPointX[num]) && (Minutiae[p].Y == curPointY[num]))
                                    {
                                        var temp = Minutiae[cur];
                                        temp.Angle = Minutiae[p].Angle;
                                        Minutiae[cur] = temp;
                                    }
                                }
                                if (((nextX != 0) || (nextY != 0)) && (!StopSteps[num]))
                                {
                                    prevPointX[num] = curPointX[num];
                                    prevPointY[num] = curPointY[num];
                                    curPointX[num] = nextX;
                                    curPointY[num] = nextY;
                                }
                            }
                        }
                }
                int maxNum = 0;
                if (!Find)
                {
                    int max = Depth[0];
                    if (max < Depth[1])
                    {
                        max = Depth[1];
                        maxNum = 1;
                    }
                    if (max < Depth[2])
                    {
                        max = Depth[2];
                        maxNum = 2;
                    }
                }
                double maxAngle = Math.Acos((((double)(startPointY[maxNum] - curY)) / Math.Sqrt((double)((startPointY[maxNum] - curY) * (startPointY[maxNum] - curY) + (startPointX[maxNum] - curX) * (startPointX[maxNum] - curX)))));
                if ((startPointX[maxNum] - curX) > 0)
                    maxAngle = 2 * Math.PI - maxAngle;
                if (Math.Abs(maxAngle - Teta) < Math.PI/2)
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

        private static void FindDirectionOfBifurcationUsingQuadrantsAndAlignment(int cur, double Teta,
                                                                                 List<Minutia> Minutiae,
                                                                                 int[,] BinaryImage)
        {
            int radius = 3; //Use squares 7x7
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            double score1 = 0;
            double score2 = 0;
            double angle = 0;
            double number = 0;
            if (Teta < Math.PI/2) //Quadrant I or III
            {
                for (int i = 0; i <= radius; i++) //Check Quadrant III
                    for (int j = 0; j >= (-1)*radius; j--)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    if (angle < 0)
                                        angle += Math.PI;
                                    score1 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score1 = Math.PI * 2;
                else
                    score1 = score1 / number;
                number = 0;
                for (int i = 0; i >= (-1)*radius; i--) //Check Quadrant I
                    for (int j = 0; j <= radius; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    if (angle < 0)
                                        angle += Math.PI;
                                    score2 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score2 = Math.PI * 2;
                else
                    score2 = score2 / number;
                if (score1 < score2)
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
            else   //Quadrant II or IV
            {
                for (int i = 0; i <= radius; i++) //Check Quadrant IV
                    for (int j = 0; j <= radius; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    if (angle < 0)
                                        angle += Math.PI;
                                    score1 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score1 = Math.PI * 2;
                else
                    score1 = score1 / number;
                number = 0;
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant II
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1)*i/Math.Abs(i)*Math.PI/2;
                                    else
                                        angle = Math.Atan((double) (-1)*i/(double) j);
                                    if (angle < 0)
                                        angle += Math.PI;
                                    score2 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score2 = Math.PI*2;
                else
                    score2 = score2 / number;
                if (score1 < score2)
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

        public static void FindDirection(double[,] OrientationField, int dim, List<Minutia> Minutiae , int[,] BinaryImage,int Version)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count == 2) //Ending of Line
                {
                    double angle = 0;
                    for (int i = -1; i < 2; i++)
                        for (int j = -1; j < 2; j++)
                            if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                                (curY + j < BinaryImage.GetLength(1)))
                                if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                                {
                                    angle = Math.Acos(((double) j)/Math.Sqrt((double) (i*i + j*j)));
                                    if (i > 0)
                                        angle = 2 * Math.PI - angle;
                                }
                    if ((Teta - angle < Math.PI/2) && (angle - Teta < Math.PI/2))
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
                else //Bifurcation
                    switch (Version)
                    {
                        case 1:
                            FindDirectionOfBifurcationUsingQuadrants(cur, Teta, Minutiae, BinaryImage);
                            break;
                        case 2:
                            FindDirectionOfBifurcationUsingAverageValues(cur,Teta,Minutiae,BinaryImage);
                            break;
                        case 3:
                            FindDirectionOfBifurcationUsingAlignment(cur,Teta,Minutiae,BinaryImage,Math.PI/10,5);
                            break;
                        case 4:
                            FindDirectionOfBifurcationUsingQuadrantsAndAlignment(cur, Teta, Minutiae, BinaryImage);
                            break;
                        default:
                            break;
                    }
            }
        }


        public static void FindDirectionVersion2(double[,] OrientationField, int dim, List<Minutia> Minutiae, int[,] BinaryImage)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count == 2) //Ending of Line
                {
                    double angle = 0;
                    for (int i = -1; i < 2; i++)
                        for (int j = -1; j < 2; j++)
                            if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                                (curY + j < BinaryImage.GetLength(1)))
                                if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                                {
                                    angle = Math.Acos(((double) j)/Math.Sqrt((double) (i*i + j*j)));
                                    if (i > 0)
                                        angle = 2*Math.PI - angle;
                                }
                    if ((Teta - angle < Math.PI/2) && (angle - Teta < Math.PI/2))
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
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count > 2) //Bifurcation
                    FindDirectionOfBifurcationUsingAlignment(cur, Teta, Minutiae, BinaryImage, Math.PI / 10, 5);
            }
        }*/
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
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
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
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
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
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
                                        minAngle = Math.Abs(angle - Teta);
                                    }
                                    else
                                    {
                                        angle = Math.Acos(((double)j) / Math.Sqrt(((double)(i * i + j * j))));
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

        private static void CheckPath(int curX, int curY, int prevX, int prevY, int[,] BinaryImage,ref double score,ref
                                        int number,int radiusOfSearch)
        {
            bool StopSearch = false;
            int nextX = curX;
            int nextY = curY;
            number++;
            double angle = Math.Acos((((double)(curY-prevY))/Math.Sqrt((double)((curY-prevY)*(curY-prevY) + (curX-prevX)*(curX-prevX)))));
            if ((curX - prevX) > 0)
                angle = 2*Math.PI - angle;
            score += angle;
            if (number < radiusOfSearch)
                for (int i = -1; (i < 2) && (!StopSearch); i++)
                    for (int j = -1; (j < 2) && (!StopSearch); j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                        (curY + j < BinaryImage.GetLength(1)))
                            if (((i != 0) || (j != 0)) && ((curX + i != prevX) || (curY + j != prevY)))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                    if ((nextX == curX) && (nextY == curY))
                                    {
                                        nextX = curX + i;
                                        nextY = curY + j;
                                    }
                                    else
                                        StopSearch = true;
            if ((nextX == curX) && (nextY == curY))
                StopSearch = true;
            if (!StopSearch)
                CheckPath(nextX, nextY, curX, curY, BinaryImage,ref score,ref number, radiusOfSearch);
        }

        private static void FindDirectionOfBifurcationUsingAverageValues(int cur, double Teta, List<Minutia> Minutiae,
                                                                         int[,] BinaryImage)
        {
            int RadiusOfSearch = 7;
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            double minAngle = 2*Math.PI;
            int minI = 0;
            int minJ = 0;
            for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                    if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                        (curY + j < BinaryImage.GetLength(1)))
                        if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                        {
                            int number = 0;
                            double score = 0;
                            CheckPath(curX + i, curY + j, curX, curY, BinaryImage,ref score,ref number, RadiusOfSearch);
                            double angle = score/number;
                            if (angle > Math.PI)
                                angle = +Math.PI;
                            if (Math.Abs(angle - Teta) < minAngle)
                            {
                                minI = i;
                                minJ = j;
                                minAngle = angle;
                            }
                        }
            if ((minJ <= 0) && (minI >= 0))
            {
                var temp = Minutiae[cur];
                temp.Angle = Teta;
                Minutiae[cur] = temp;
            }
            else
            {
                var temp = Minutiae[cur];
                temp.Angle = Teta + Math.PI;
                Minutiae[cur] = temp;
            }      
        }
        

        private static void FindDirectionOfBifurcationUsingAlignment(int cur, double Teta, List<Minutia> Minutiae,
                                                                     int[,] BinaryImage,double delta,int step)
        {
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            if ((curX > 0) && (curX < BinaryImage.GetLength(0)-1) && (curY < BinaryImage.GetLength(1)-1) && (curY > 0)) {
                int[] startPointX = new int[3];
                int[] startPointY = new int[3];
                int[] curPointX = new int[3];
                int[] curPointY = new int[3];
                int[] prevPointX = new int[3];
                int[] prevPointY = new int[3];
                bool[] StopSteps = new bool[3];
                int[] Depth = new int[3];
                int num = 0;
                for (int i = -1 ; i < 2;i ++)
                    for (int j = -1 ; j < 2; j++)
                        if (((j != 0) || (i != 0)) && (BinaryImage[curX + i,curY+  j] == 0))
                        {
                            startPointX[num] = curX + i;
                            startPointY[num] = curY + j;
                            curPointX[num] = curX + i;
                            curPointY[num] = curY + j;
                            prevPointX[num] = curX;
                            prevPointY[num] = curY;
                            num++;
                        }
                StopSteps[0] = false;
                StopSteps[1] = false;
                StopSteps[2] = false;
                Depth[0] = 0;
                Depth[1] = 0;
                Depth[2] = 0;
                int number = 0;
                double score = 0;
                bool Find = false;
                while (((!StopSteps[0]) && (!StopSteps[1]) && (!StopSteps[2])) && (Depth[0] + Depth[1] + Depth[2] < 50) && (!Find))
                {
                    for (num = 0; num <3 ; num++)
                        if (!StopSteps[num])
                        {
                            score = 0;
                            number = 0;
                            for (int k = 0; (k < step) && (!StopSteps[num]); k++)
                            {
                                int nextX = 0;
                                int nextY = 0;
                                Depth[num]++;
                                number++;
                                double angle = Math.Acos((((double)(curPointY[num]-prevPointY[num]))/Math.Sqrt((double)((curPointY[num]-prevPointY[num])*(curPointY[num]-prevPointY[num]) + (curPointX[num]-prevPointX[num])*(curPointX[num]-prevPointX[num])))));
                                if ((curPointX[num] - prevPointX[num]) > 0)
                                    angle = 2*Math.PI - angle;
                                score += angle;
                                if (Math.Abs(score/num - Teta) < delta)
                                    StopSteps[num] = true;
                                for (int i = -1; (i < 2) && (!StopSteps[num]); i++)
                                    for (int j = -1; (j < 2) && (!StopSteps[num]); j++)
                                    if ((curPointX[num] + i >= 0) && (curPointX[num] + i < BinaryImage.GetLength(0)) && (curPointY[num] + j >= 0) &&
                                        (curPointY[num] + j < BinaryImage.GetLength(1)))
                                        if (((i != 0) || (j != 0)) &&
                                            (BinaryImage[curPointX[num] + i, curPointY[num] + j] == 0) &&
                                            ((curPointX[num] + i != prevPointX[num]) ||
                                         (curPointY[num] + j != prevPointY[num])))
                                        {
                                            if ((nextX == 0) && (nextY == 0))
                                            {
                                                nextX = i;
                                                nextY = j;
                                            }
                                            else
                                                StopSteps[num] = true;
                                        }
                                if ((nextX == 0) && (nextY == 0) && (!StopSteps[num]))
                                {
                                    Find = true;
                                    int p = 0;
                                    while (((Minutiae[p].X != curPointX[num]) || (Minutiae[p].Y != curPointY[num])) && (p<Minutiae.Count()-1))
                                        p++;
                                    if ((Minutiae[p].X == curPointX[num]) && (Minutiae[p].Y == curPointY[num]))
                                    {
                                        var temp = Minutiae[cur];
                                        temp.Angle = Minutiae[p].Angle;
                                        Minutiae[cur] = temp;
                                    }
                                }
                                if (((nextX != 0) || (nextY != 0)) && (!StopSteps[num]))
                                {
                                    prevPointX[num] = curPointX[num];
                                    prevPointY[num] = curPointY[num];
                                    curPointX[num] = nextX;
                                    curPointY[num] = nextY;
                                }
                            }
                        }
                }
                int maxNum = 0;
                if (!Find)
                {
                    int max = Depth[0];
                    if (max < Depth[1])
                    {
                        max = Depth[1];
                        maxNum = 1;
                    }
                    if (max < Depth[2])
                    {
                        max = Depth[2];
                        maxNum = 2;
                    }
                }
                double maxAngle = Math.Acos((((double)(startPointY[maxNum] - curY)) / Math.Sqrt((double)((startPointY[maxNum] - curY) * (startPointY[maxNum] - curY) + (startPointX[maxNum] - curX) * (startPointX[maxNum] - curX)))));
                if ((startPointX[maxNum] - curX) > 0)
                    maxAngle = 2 * Math.PI - maxAngle;
                if (Math.Abs(maxAngle - Teta) < Math.PI/2)
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

        private static void FindDirectionOfBifurcationUsingQuadrantsAndAlignment(int cur, double Teta,
                                                                                 List<Minutia> Minutiae,
                                                                                 int[,] BinaryImage)
        {
            int radius = 3; //Use squares 7x7
            int curX = Minutiae[cur].Y;
            int curY = Minutiae[cur].X;
            double score1 = 0;
            double score2 = 0;
            double angle = 0;
            double number = 0;
            if (Teta >= 0) //Quadrant I or III
            {
                for (int i = 0; i <= radius; i++) //Check Quadrant III
                    for (int j = 0; j >= (-1)*radius; j--)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score1 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score1 = Math.PI * 2;
                else
                    score1 = score1 / number;
                number = 0;
                for (int i = 0; i >= (-1)*radius; i--) //Check Quadrant I
                    for (int j = 0; j <= radius; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score2 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score2 = Math.PI * 2;
                else
                    score2 = score2 / number;
                if (score1 < score2)
                {
                    var temp = Minutiae[cur];
                    temp.Angle = Teta - Math.PI;
                    Minutiae[cur] = temp;
                }
                else
                {
                    var temp = Minutiae[cur];
                    temp.Angle = Teta;
                    Minutiae[cur] = temp;
                }
            }
            else   //Quadrant II or IV
            {
                for (int i = 0; i <= radius; i++) //Check Quadrant IV
                    for (int j = 0; j <= radius; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score1 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score1 = Math.PI * 2;
                else
                    score1 = score1 / number;
                number = 0;
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant II
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curX + i, curY + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1)*i/Math.Abs(i)*Math.PI/2;
                                    else
                                        angle = Math.Atan((double) (-1)*i/(double) j);
                                    score2 += Math.Abs(angle - Teta);
                                }
                if (number == 0)
                    score2 = Math.PI*2;
                else
                    score2 = score2 / number;
                if (score1 < score2)
                {
                    var temp = Minutiae[cur];
                    temp.Angle = Teta ;
                    Minutiae[cur] = temp;
                }
                else
                {
                    var temp = Minutiae[cur];
                    temp.Angle = Teta + Math.PI;
                    Minutiae[cur] = temp;
                }
            }
        }

        public static void FindDirection(double[,] OrientationField, int dim, List<Minutia> Minutiae , int[,] BinaryImage,int Version)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count == 2) //Ending of Line
                {
                    double angle = 0;
                    for (int i = -1; i < 2; i++)
                        for (int j = -1; j < 2; j++)
                            if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                                (curY + j < BinaryImage.GetLength(1)))
                                if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                                {
                                    angle = Math.Atan2((-1) * i, j);
                                }
                    if (Teta >= 0)
                    {
                        if (Math.Abs(Teta - angle) < Math.PI/2)
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Teta - Math.PI;
                            Minutiae[cur] = temp;
                        }
                        else
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Teta;
                            Minutiae[cur] = temp;
                        }
                    }
                    else
                    {
                        if (Math.Abs(Teta - angle) < Math.PI / 2)
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
                else //Bifurcation
                    switch (Version)
                    {
                        case 1:
                            FindDirectionOfBifurcationUsingQuadrants(cur, Teta, Minutiae, BinaryImage);
                            break;
                        case 2:
                            FindDirectionOfBifurcationUsingAverageValues(cur,Teta,Minutiae,BinaryImage);
                            break;
                        case 3:
                            FindDirectionOfBifurcationUsingAlignment(cur,Teta,Minutiae,BinaryImage,Math.PI/10,5);
                            break;
                        case 4:
                            FindDirectionOfBifurcationUsingQuadrantsAndAlignment(cur, Teta, Minutiae, BinaryImage);
                            break;
                        default:
                            break;
                    }
            }
        }


        public static void FindDirectionVersion2(double[,] OrientationField, int dim, List<Minutia> Minutiae, int[,] BinaryImage)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count == 2) //Ending of Line
                {
                    double angle = 0;
                    for (int i = -1; i < 2; i++)
                        for (int j = -1; j < 2; j++)
                            if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                                (curY + j < BinaryImage.GetLength(1)))
                                if (((i != 0) || (j != 0)) && (BinaryImage[curX + i, curY + j] == 0))
                                {
                                    angle = Math.Acos(((double) j)/Math.Sqrt((double) (i*i + j*j)));
                                    if (i > 0)
                                        angle = 2*Math.PI - angle;
                                }
                    if ((Teta - angle < Math.PI/2) && (angle - Teta < Math.PI/2))
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
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Teta = OrientationField[curX/dim, curY/dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count > 2) //Bifurcation
                    FindDirectionOfBifurcationUsingAlignment(cur, Teta, Minutiae, BinaryImage, Math.PI / 10, 5);
            }
        }
    }
}
