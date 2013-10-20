using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinning
{
    public static class MinutiaeDirection
    {
        private static double FindDirectionOfBifurcationUsingQuadrants(int curI, int curJ, double Theta, int[,] BinaryImage)
        {
            double result = 0;
            int radius = 3; //Use squares 7x7
            int minI = 0;
            int minJ = 0;
            double minAngle = 0;
            double angle = 0;
            if (Theta >= 0) //Step 1
            {   //Quadrant I or III
                //Step 2
                for (int i = 0; i <= radius; i++) //Check Quadrant III
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) && (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                    if ((minI == 0) && (minJ == 0))
                                    {
                                        minI = i;
                                        minJ = j;
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        minAngle = Math.Abs(angle - Theta);
                                    }
                                    else
                                    {
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        if (Math.Abs(angle - Theta) < minAngle)
                                        {
                                            minAngle = Math.Abs(angle - Theta);
                                            minI = i;
                                            minJ = j;
                                        }
                                    }


                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant I
                    for (int j = 0; j <= radius; j++)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) && (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                    if ((minI == 0) && (minJ == 0))
                                    {
                                        minI = i;
                                        minJ = j;
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        minAngle = Math.Abs(angle - Theta);
                                    }
                                    else
                                    {
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        if (Math.Abs(angle - Theta) < minAngle)
                                        {
                                            minAngle = Math.Abs(angle - Theta);
                                            minI = i;
                                            minJ = j;
                                        }
                                    }
                if ((minI >= 0) && (minJ <= 0)) //Steps 3А, 3Б
                    result = Theta - Math.PI;
                else
                    result = Theta;

            }
            else //Quadrant II or IV
            {
                for (int i = 0; i <= radius; i++) //Check Quadrant  IV
                    for (int j = 0; j <= radius; j++)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) && (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                    if ((minI == 0) && (minJ == 0))
                                    {
                                        minI = i;
                                        minJ = j;
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        minAngle = Math.Abs(angle - Theta);
                                    }
                                    else
                                    {
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        if (Math.Abs(angle - Theta) < minAngle)
                                        {
                                            minAngle = Math.Abs(angle - Theta);
                                            minI = i;
                                            minJ = j;
                                        }
                                    }
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant  II
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) && (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                    if ((minI == 0) && (minJ == 0))
                                    {
                                        minI = i;
                                        minJ = j;
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        minAngle = Math.Abs(angle - Theta);
                                    }
                                    else
                                    {
                                        angle = Math.Atan((double)((-1) * i) / j);
                                        if (Math.Abs(angle - Theta) < minAngle)
                                        {
                                            minAngle = Math.Abs(angle - Theta);
                                            minI = i;
                                            minJ = j;
                                        }
                                    }
                if ((minI >= 0) && (minJ >= 0)) //Steps 3В, 3Г
                    result = Theta;
                else
                    result = Theta + Math.PI;
            }
            return result;
        }

        private static void GetScoreFromPath(int curI, int curJ, int prevI, int prevJ, int[,] BinaryImage, ref double score, ref
                                        int number, int radiusOfSearch)
        {
            bool StopSearch = false;
            int nextI = curI;
            int nextJ = curJ;
            number++;
            double angle = Math.Atan2((-1) * (curI - prevI), curJ - prevJ);
            score += angle;
            if (number < radiusOfSearch)
                for (int i = -1; (i < 2) && (!StopSearch); i++)
                    for (int j = -1; (j < 2) && (!StopSearch); j++)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) &&
                        (curJ + j < BinaryImage.GetLength(1)))
                            if (((i != 0) || (j != 0)) && ((curI + i != prevI) || (curJ + j != prevJ)))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                    if ((nextI == curI) && (nextJ == curJ))
                                    {
                                        nextI = curI + i;
                                        nextJ = curJ + j;
                                    }
                                    else
                                        StopSearch = true;
            if ((nextI == curI) && (nextJ == curJ))
                StopSearch = true;
            if (!StopSearch)
                GetScoreFromPath(nextI, nextJ, curI, curJ, BinaryImage, ref score, ref number, radiusOfSearch);
        }

        private static double FindDirectionOfBifurcationUsingPathsAndAverageValues(int curI, int curJ, double Theta, List<Point> locality,
                                                                         int[,] BinaryImage)
        {
            double result = 0;
            int RadiusOfSearch = 7;
            double minAngle = 2 * Math.PI;
            int minI = 0;
            int minJ = 0;
            for (int i = 0; i < locality.Count; i++)
            {
                int number = 0;
                double score = 0;
                GetScoreFromPath(curI + locality[i].X, curJ + locality[i].Y, curI, curJ, BinaryImage, ref score, ref number, RadiusOfSearch); //Step 1
                double angle = score / number; //Step 2
                if (angle > Math.PI / 2)
                    angle -= Math.PI;
                if (angle < (-1) * Math.PI / 2)
                    angle += Math.PI;
                if (Math.Abs(angle - Theta) < minAngle) //Step 3
                {
                    minI = locality[i].X; //Step 4
                    minJ = locality[i].Y;
                    minAngle = angle;
                }
            }
            if (minJ < 0)  //Step 5
            {
                if (Theta >= 0)
                    result = Theta - Math.PI;
                else
                    result = Theta + Math.PI;
            }
            else
                result = Theta;
            return result;
        }


        private static double FindDirectionOfBifurcationUsingAlignment(int curI, int curJ, double Theta,
                                                                     int[,] BinaryImage, List<Point> locality, double delta, int step)
        {
            double result = 0;
                int[] startPointI = new int[3];
                int[] startPointJ = new int[3];
                int[] curPointI = new int[3];
                int[] curPointJ = new int[3];
                int[] prevPointI = new int[3];
                int[] prevPointJ = new int[3];
                bool[] StopSteps = new bool[3];
                int[] Depth = new int[3];
                for (int num = 0; num < 3; num++)
                {
                    startPointI[num] = curI + locality[num].X;
                    startPointJ[num] = curJ + locality[num].Y;
                    curPointI[num] = curI + locality[num].X;
                    curPointJ[num] = curJ + locality[num].Y;
                    prevPointI[num] = curI;
                    prevPointJ[num] = curJ;
                }
                StopSteps[0] = false;
                StopSteps[1] = false;
                StopSteps[2] = false;
                Depth[0] = 0;
                Depth[1] = 0;
                Depth[2] = 0;
                int number = 0;
                double score = 0;
                int numberOfSteps = 0;
                while (((!StopSteps[0]) && (!StopSteps[1]) && (!StopSteps[2])) && (numberOfSteps < 10)) //Step 2
                {
                    numberOfSteps++;
                    for (int num = 0; num < 3; num++)
                        if (!StopSteps[num])
                        {
                            score = 0;
                            number = 0;
                            for (int k = 0; (k < step) && (!StopSteps[num]); k++)
                            {
                                int nextI = 0;
                                int nextJ = 0;
                                Depth[num]++;
                                number++;
                                double angle = Math.Atan2(((double)(prevPointI[num] - curPointI[num])),
                                                          (curPointJ[num] - prevPointJ[num]));
                                score += angle;
                                double resScore = score / number;
                                if (resScore > Math.PI / 2)
                                    resScore = resScore - Math.PI;
                                if (resScore < (-1) * Math.PI / 2)
                                    resScore = resScore + Math.PI;
                                if (Math.Abs(score / Depth[num] - Theta) < delta)
                                    StopSteps[num] = true;
                                for (int i = -1; (i < 2) && (!StopSteps[num]); i++)
                                    for (int j = -1; (j < 2) && (!StopSteps[num]); j++)
                                        if ((curPointI[num] + i >= 0) && (curPointI[num] + i < BinaryImage.GetLength(0)) && (curPointJ[num] + j >= 0) &&
                                            (curPointJ[num] + j < BinaryImage.GetLength(1)))
                                            if (((i != 0) || (j != 0)) &&
                                                (BinaryImage[curPointI[num] + i, curPointJ[num] + j] == 0) &&
                                                ((curPointI[num] + i != prevPointI[num]) ||
                                             (curPointJ[num] + j != prevPointJ[num])))
                                            {
                                                if ((nextI == 0) && (nextJ == 0))
                                                {
                                                    nextI = i;
                                                    nextJ = j;
                                                }
                                                else
                                                {
                                                    StopSteps[num] = true;
                                                    Depth[num] = 50;
                                                }
                                            }
                                if ((nextI == 0) && (nextJ == 0))
                                {
                                    StopSteps[num] = true;
                                    Depth[num] = 50;
                                }
                                if (((nextI != 0) || (nextJ != 0)) && (!StopSteps[num]))
                                {
                                    prevPointI[num] = curPointI[num];
                                    prevPointJ[num] = curPointJ[num];
                                    curPointI[num] = nextI;
                                    curPointJ[num] = nextJ;
                                }
                            }
                        }
                }
                int maxNum = 0;   //Step 3
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
                double maxAngle = Math.Atan2(curI - startPointI[maxNum], startPointJ[maxNum] - curJ); //Step 4
                if (Math.Abs(maxAngle - Theta) < Math.PI/2)
                {
                    if (Theta >= 0)
                        result = Theta - Math.PI;
                    else
                        result = Theta + Math.PI;
                }
                else
                    result = Theta;
            return result;
        }

        private static double FindDirectionOfBifurcationUsingQuadrantsAndAverageValues(int curI, int curJ, double Theta,
                                                                                 int[,] BinaryImage)
        {
            double result = 0;
            int radius = 3; //Use squares 7x7
            double score1 = 0;
            double score2 = 0;
            double angle = 0;
            double number = 0;
            if (Theta >= 0) //Step 1
            {//Quadrant I or III
                //Step 2
                for (int i = 0; i <= radius; i++) //Check Quadrant III
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) &&
                            (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score1 += Math.Abs(angle - Theta);
                                }
                if (number == 0)
                    score1 = Math.PI * 2;
                else
                    score1 = score1 / number;
                number = 0;
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant I
                    for (int j = 0; j <= radius; j++)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) &&
                            (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score2 += Math.Abs(angle - Theta);
                                }
                if (number == 0)
                    score2 = Math.PI * 2;
                else
                    score2 = score2 / number;
                if (score1 < score2) //Steps 3,4
                    result = Theta - Math.PI;
                else
                    result = Theta;
            }
            else   //Quadrant II or IV
            {
                //Step 2
                for (int i = 0; i <= radius; i++) //Check Quadrant IV
                    for (int j = 0; j <= radius; j++)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) &&
                            (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score1 += Math.Abs(angle - Theta);
                                }
                if (number == 0)
                    score1 = Math.PI * 2;
                else
                    score1 = score1 / number;
                number = 0;
                for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant II
                    for (int j = 0; j >= (-1) * radius; j--)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) &&
                            (curJ + j < BinaryImage.GetLength(1)))
                            if ((i != 0) || (j != 0))
                                if (BinaryImage[curI + i, curJ + j] == 0)
                                {
                                    number++;
                                    if (j == 0)
                                        angle = (-1) * i / Math.Abs(i) * Math.PI / 2;
                                    else
                                        angle = Math.Atan((double)(-1) * i / (double)j);
                                    score2 += Math.Abs(angle - Theta);
                                }
                if (number == 0)
                    score2 = Math.PI * 2;
                else
                    score2 = score2 / number;
                if (score1 < score2) //Steps 3,4
                    result = Theta;
                else
                    result = Theta + Math.PI;
            }
            return result;
        }

        public static void FindDirection(double[,] OrientationField, int dim, List<Minutia> Minutiae, int[,] BinaryImage, int Version)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curI = Minutiae[cur].Y;
                int curJ = Minutiae[cur].X;
                double Theta = OrientationField[curI / dim, curJ / dim];
                List<Point> locality = new List<Point>();
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curI + i >= 0) && (curI + i < BinaryImage.GetLength(0)) && (curJ + j >= 0) &&
                            (curJ + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curI + i, curJ + j] == 0)
                                if ((i != 0) || (j != 0))
                                    locality.Add(new Point(i, j));
                if (locality.Count == 1) //Ending of Line
                {
                    double angle = Math.Atan2((-1) * locality[0].X, locality[0].Y);
                    if (Theta >= 0)
                    {
                        if (Math.Abs(Theta - angle) < Math.PI / 2)
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Theta - Math.PI;
                            Minutiae[cur] = temp;
                        }
                        else
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Theta;
                            Minutiae[cur] = temp;
                        }
                    }
                    else
                    {
                        if (Math.Abs(Theta - angle) < Math.PI / 2)
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Theta + Math.PI;
                            Minutiae[cur] = temp;
                        }
                        else
                        {
                            var temp = Minutiae[cur];
                            temp.Angle = Theta;
                            Minutiae[cur] = temp;
                        }
                    }
                }
                else
                {//Bifurcation 
                    var temp = Minutiae[cur];
                    switch (Version)
                    {

                        case 1:  
                            temp.Angle = FindDirectionOfBifurcationUsingQuadrants(curI, curJ, Theta, BinaryImage);
                            break;
                        case 2:  
                            temp.Angle = FindDirectionOfBifurcationUsingPathsAndAverageValues(curI, curJ, Theta, locality, BinaryImage);
                            break;
                        case 3:
                            temp.Angle = FindDirectionOfBifurcationUsingAlignment(curI, curJ, Theta, BinaryImage, locality, Math.PI / 8, 5);
                            break;
                        case 4: 
                            temp.Angle = FindDirectionOfBifurcationUsingQuadrantsAndAverageValues(curI, curJ, Theta, BinaryImage);
                            break;
                        default:
                            break;
                    }
                    Minutiae[cur] = temp;
                }
            }
        }


        /*public static void FindDirectionVersion2(double[,] OrientationField, int dim, List<Minutia> Minutiae, int[,] BinaryImage)
        {
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Theta = OrientationField[curX / dim, curY / dim];
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
                                    angle = Math.Acos(((double)j) / Math.Sqrt((double)(i * i + j * j)));
                                    if (i > 0)
                                        angle = 2 * Math.PI - angle;
                                }
                    if ((Theta - angle < Math.PI / 2) && (angle - Theta < Math.PI / 2))
                    {
                        var temp = Minutiae[cur];
                        temp.Angle = Theta + Math.PI;
                        Minutiae[cur] = temp;
                    }
                    else
                    {
                        var temp = Minutiae[cur];
                        temp.Angle = Theta;
                        Minutiae[cur] = temp;
                    }
                }
            }
            for (int cur = 0; cur < Minutiae.Count(); cur++)
            {
                int curX = Minutiae[cur].Y;
                int curY = Minutiae[cur].X;
                double Theta = OrientationField[curX / dim, curY / dim];
                int count = 0;
                for (int i = -1; i < 2; i++)
                    for (int j = -1; j < 2; j++)
                        if ((curX + i >= 0) && (curX + i < BinaryImage.GetLength(0)) && (curY + j >= 0) &&
                            (curY + j < BinaryImage.GetLength(1)))
                            if (BinaryImage[curX + i, curY + j] == 0)
                                count++;
                if (count > 2) //Bifurcation
                    FindDirectionOfBifurcationUsingAlignment(curI,cur, Theta, Minutiae, BinaryImage, Math.PI / 10, 5);
            }
        }*/
    }
}
