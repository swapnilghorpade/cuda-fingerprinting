#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

extern "C"{
__declspec(dllexport) void FindDirection(double *OrientationField, int orHeight,int orWidth,int dim, int *Minutiae , int NoM,int *BinaryImage,int imgHeight,int imgWidth,double *Directions);
}

double FindDirectionOfBifurcationUsingQuadrantsAndAlignment(int curX,int curY, double Teta, int *BinaryImage,int imgHeight,int imgWidth)
{
    int radius = 3; //Use squares 7x7
    double score1 = 0;
    double score2 = 0;
    double angle = 0;
    int number = 0;
	double result = 0;
    if (Teta < M_PI/2) //Quadrant I or III
    {
        for (int i = 0; i <= radius; i++) //Check Quadrant III
             for (int j = 0; j >= (-1)*radius; j--)
                  if ((curX + i >= 0) && (curX + i < imgHeight) && (curY + j >= 0) && (curY + j < imgWidth))
                      if ((i != 0) || (j != 0))
                          if (BinaryImage[(curX + i) * imgWidth + curY + j] == 0)
                          {
                              number++;
                              if (j == 0)
                                  angle = (-1) * i / abs(i) * M_PI / 2;
                              else
                                  angle = atan((double)(-1) * i / (double)j);
                              if (angle < 0)
                                  angle += M_PI;
                              score1 += abs(angle - Teta);
                          }
         if (number == 0)
             score1 = M_PI * 2;
         else
             score1 = score1 / (double)number;
         number = 0;
         for (int i = 0; i >= (-1)*radius; i--) //Check Quadrant I
              for (int j = 0; j <= radius; j++)
                   if ((curX + i >= 0) && (curX + i < imgHeight) && (curY + j >= 0) && (curY + j < imgWidth))
                       if ((i != 0) || (j != 0))
                           if (BinaryImage[(curX + i) * imgWidth + curY + j] == 0)
                           {
                               number++;
                               if (j == 0)
                                   angle = (-1) * i / abs(i) * M_PI / 2;
                               else
                                   angle = atan((double)(-1) * i / (double)j);
                               if (angle < 0)
                                   angle += M_PI;
                               score2 += abs(angle - Teta);
                            }
           if (number == 0)
               score2 = M_PI * 2;
           else
               score2 = score2 / (double)number;
           if (score1 < score2)
			   result = Teta + M_PI;
           else
			   result = Teta;
     }
     else   //Quadrant II or IV
     {
         for (int i = 0; i <= radius; i++) //Check Quadrant IV
             for (int j = 0; j <= radius; j++)
                 if ((curX + i >= 0) && (curX + i < imgHeight) && (curY + j >= 0) && (curY + j < imgHeight))
                     if ((i != 0) || (j != 0))
                         if (BinaryImage[(curX + i) * imgWidth + curY + j] == 0)
                         {
                             number++;
                             if (j == 0)
                                 angle = (-1) * i / abs(i) * M_PI / 2;
                             else
                                 angle = atan((double)(-1) * i / (double)j);
                             if (angle < 0)
                                 angle += M_PI;
                             score1 += abs(angle - Teta);
                         }
          if (number == 0)
              score1 = M_PI * 2;
          else
              score1 = score1 / (double)number;
          number = 0;
          for (int i = 0; i >= (-1) * radius; i--) //Check Quadrant II
               for (int j = 0; j >= (-1) * radius; j--)
                   if ((curX + i >= 0) && (curX + i < imgHeight) && (curY + j >= 0) && (curY + j < imgWidth))
                       if ((i != 0) || (j != 0))
                           if (BinaryImage[(curX + i) * imgWidth + curY + j] == 0)
                           {
                               number++;
                               if (j == 0)
                                   angle = (-1)*i/abs(i)*M_PI/2;
                               else
                                   angle = atan((double) (-1)*i/(double) j);
                               if (angle < 0)
                                   angle += M_PI;
                               score2 += abs(angle - Teta);
                           }
          if (number == 0)
              score2 = M_PI*2;
          else
              score2 = score2 / (double)number;
          if (score1 < score2)
			result = Teta + M_PI;
          else
            result = Teta;
    }
	return result;
}

void FindDirection(double *OrientationField, int orHeight,int orWidth,int dim, int *Minutiae , int NoM,int *BinaryImage,int imgHeight,int imgWidth,double *Directions)
{
    for (int cur = 0; cur < NoM; cur++)
    {
        int curX = Minutiae[2*cur+1];
        int curY = Minutiae[2*cur];
        double Teta = OrientationField[(curX/dim) * orWidth + curY/dim];
        int count = 0;
        for (int i = -1; i < 2; i++)
            for (int j = -1; j < 2; j++)
                if ((curX + i >= 0) && (curX + i < imgHeight) && (curY + j >= 0) && (curY + j < imgWidth))
                    if (BinaryImage[(curX + i) * imgWidth + curY + j] == 0)
                        count++;
        if (count == 2) //Ending of Line
        {
            double angle = 0;
            for (int i = -1; i < 2; i++)
                 for (int j = -1; j < 2; j++)
                      if ((curX + i >= 0) && (curX + i < imgHeight) && (curY + j >= 0) && (curY + j < imgWidth))
                          if (((i != 0) || (j != 0)) && (BinaryImage[(curX + i) * imgWidth + curY + j] == 0))
                          {
                              angle = acos(((double) j)/sqrt((double) (i*i + j*j)));
                              if (i > 0)
                                  angle = 2 * M_PI - angle;
                          }
             if ((Teta - angle < M_PI/2) && (angle - Teta < M_PI/2))
				Directions[cur] = Teta + M_PI;
             else
				Directions[cur] = Teta;
             }
             else //Bifurcation
                 Directions[cur] = FindDirectionOfBifurcationUsingQuadrantsAndAlignment(curX,curY,Teta, BinaryImage,imgHeight,imgWidth);
    }
}
