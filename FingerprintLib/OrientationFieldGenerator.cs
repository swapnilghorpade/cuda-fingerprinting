using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;

namespace FingerprintLib
{
    /// <summary>
    /// Составление поля направлений оператором Собеля
    /// </summary>
    public class OrientationFieldGenerator
    {
        /// <summary>
        /// Размер блока поля направлений
        /// </summary>
        public const int W = 10;

        public static int[,] GenerateXGradients(int[,] bytes)
        {
            int maxX = bytes.GetUpperBound(0)+1;
            int maxY = bytes.GetUpperBound(1)+1;
            int[,] result = new int[bytes.GetUpperBound(0)+1,bytes.GetUpperBound(1)+1];
            var filter = new int[,] {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};

                for (int y = 0; y < maxY; y++)
                {
                for (int x = 0; x < maxX; x++)
                {

                        for (int dx = -1; dx < 2; dx++)
                        {
                            for (int dy = -1; dy < 2; dy++)
                            {
                                if (x + dx < 0 || x + dx >= maxX || y + dy < 0 || y + dy >= maxY) continue;
                                result[x, y] += filter[1 + dx, 1 + dy]*bytes[dx + x, dy + y];
                            }
                        }
                        

                    }
                }

            return result;
        }

        public static int[,] GenerateYGradients(int[,] bytes)
        {
            int maxX = bytes.GetUpperBound(0) + 1;
            int maxY = bytes.GetUpperBound(1) + 1;
            int[,] result = new int[bytes.GetUpperBound(0) + 1,bytes.GetUpperBound(1) + 1];
            var filter = new int[,] {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
            ;
                for (int y = 0; y < maxY; y++)
                {
                for (int x = 0; x < maxX; x++)
                {

                        for (int dx = -1; dx < 2; dx++)
                        {
                            for (int dy = -1; dy < 2; dy++)
                            {
                                if (x + dx < 0 || x + dx >= maxX || y + dy < 0 || y + dy >= maxY) continue;
                                result[x, y] += filter[1 + dx, 1 + dy]*bytes[dx + x, dy + y];
                            }
                        }
                        

                    }
                }
                
            return result;
        }

        public static double[,] GenerateOrientationField(int[,] bytes)
        {
            var dx = GenerateXGradients(bytes);
            var dy = GenerateYGradients(bytes);
            int maxX = bytes.GetUpperBound(0);
            int maxY = bytes.GetUpperBound(1);
            int maxResultX = (bytes.GetUpperBound(0) + 1)/(W - 1);
            int maxResultY = (bytes.GetUpperBound(1) + 1)/(W - 1);
            double[,] result = new double[maxResultX,maxResultY];
            double[,] vx = new double[maxResultX,maxResultY];
            double[,] vy = new double[maxResultX,maxResultY];
                    for (int y = 0; y < maxResultY; y++)
                    {
                        for (int x = 0; x < maxResultX; x++)
                        {

                            for (int u = 0; u < W; u++)
                            {
                                for (int v = 0; v < W; v++)
                                {
                                    int mX = x*(W - 1) + u;
                                    int mY = y*(W - 1) + v;
                                    if (mX > dx.GetUpperBound(0) || mY > dy.GetUpperBound(1)) continue;
                                    vx[x, y] += 2.0*(double) dx[mX, mY]*dy[mX, mY];
                                    vy[x, y] += -(double) dx[mX, mY]*dx[mX, mY] +
                                                (double) dy[mX, mY]*dy[mX, mY];
                                }
                            }


                            var hypotenuse = Math.Sqrt(vy[x, y] * vy[x, y] + vx[x, y] * vx[x, y]);
                             vx[x, y] = vx[x, y] / hypotenuse;
                            //float xx = (float) vx[x, y];
                            vy[x, y] = vy[x, y] / hypotenuse;
                            //float yy = (float)vy[x, y];
                        }
                 
            }
            
                for (int y = 0; y < maxResultY; y++)
                {
                    for (int x = 0; x < maxResultX; x++)
                    {
                        double resultX = 0, resultY = 0;
                        int count = 0;
                        for (int i = -1; i < 2; i++)
                        {
                            if (x + i < 0 || x + i >= maxResultX) continue;
                            for (int j = -1; j < 2; j++)
                            {
                                if (y + j < 0 || y + j >= maxResultY) continue;
                                resultX += vx[x + i, y + j];
                                resultY += vy[x + i, y + j];
                                count++;
                            }
                        }
                        //var xx = vy[x, y];
                        //var yy = vx[x, y];
                        var xx = resultY/count;
                        var yy = resultX/count;
                        if (vy[x, y] == 0 && vx[x, y] == 0) result[x, y] = double.NaN;
                        else
                        {
                            if (xx > 0 && yy >= 0)
                                result[x, y] = Math.Atan(yy/xx);
                            if (xx > 0 && yy < 0)
                                result[x, y] = Math.Atan(yy/xx) + 2*Math.PI;
                            if (xx < 0)
                                result[x, y] = Math.Atan(yy/xx) + Math.PI;
                            if (xx == 0 && yy > 0)
                                result[x, y] = Math.PI/2;
                            if (xx == 0 && yy < 0)
                                result[x, y] = 3*Math.PI/2;
                            result[x, y] = result[x, y]/2 + Math.PI/2;
                            if (result[x, y] > Math.PI) result[x, y] -= Math.PI;
                            if (double.IsNaN(xx) || double.IsNaN(yy)) result[x, y] = double.NaN;
                        }
                        
                    }
                }
                
            //result = SmoothOrientationField(result);
            //SaveField(result,"C:\\temp\\orfield.bmp");
            return result;
        }

        //public static double[,] SmoothOrientationField2(double[,] field)
        //{
        //    var FiX = new double[field.GetUpperBound(0) + 1,field.GetUpperBound(1) + 1];
        //    var FiY = new double[field.GetUpperBound(0) + 1,field.GetUpperBound(1) + 1];
        //    for (int x = 0; x <= field.GetUpperBound(0); x++)
        //    {
        //        for (int y = 0; y <= field.GetUpperBound(1); y++)
        //        {
        //            FiX[x, y] = Math.Cos(2*field[x, y]);
        //            FiY[x, y] = Math.Sin(2*field[x, y]);
        //        }
        //    }
        //    var FiXsmoothed = new double[field.GetUpperBound(0) + 1,field.GetUpperBound(1) + 1];
        //    var FiYsmoothed = new double[field.GetUpperBound(0) + 1,field.GetUpperBound(1) + 1];
        //    var fieldSmoothed = new double[field.GetUpperBound(0) + 1,field.GetUpperBound(1) + 1];
        //    for (int y = 0; y <= field.GetUpperBound(1); y++)
        //    {
        //        for (int x = 0; x <= field.GetUpperBound(0); x++)
        //        {

        //            double resultX = 0, resultY = 0;
        //            int count = 0;
        //            for (int i = -1; i < 2; i++)
        //            {
        //                if (x + i < 0 || x + i > field.GetUpperBound(0)) continue;
        //                for (int j = -1; j < 2; j++)
        //                {
        //                    if (y + j < 0 || y + j > field.GetUpperBound(1)) continue;
        //                    resultX += FiX[x + i, y + j];
        //                    resultY += FiY[x + i, y + j];
        //                    count++;
        //                }
        //            }
        //            FiXsmoothed[x, y] = resultX/count;
        //            FiYsmoothed[x, y] = resultY/count;
        //            // бага тут
        //            // = 0.5*Math.Atan(FiYsmoothed[x, y]/FiXsmoothed[x, y]);
        //            var xx = FiXsmoothed[x, y];
        //            var yy = FiYsmoothed[x, y];
        //            if (xx > 0 && yy >= 0)
        //                fieldSmoothed[x, y] = Math.Atan(yy/xx)/2;
        //            if (xx > 0 && yy < 0)
        //                fieldSmoothed[x, y] = Math.Atan(yy/xx)/2 + Math.PI;
        //            if (xx < 0)
        //                fieldSmoothed[x, y] = Math.Atan(yy/xx)/2 + Math.PI/2;
        //            if (xx == 0 && yy > 0)
        //                fieldSmoothed[x, y] = Math.PI/4;
        //            if (xx == 0 && yy < 0)
        //                fieldSmoothed[x, y] = 3*Math.PI/4;
        //            if (double.IsNaN(xx) || double.IsNaN(yy)) fieldSmoothed[x, y] = double.NaN;
        //        }
        //    }
        //    return fieldSmoothed;
        //}

        public static double[,] SmoothOrientationField(double[,] field)
        {
            var FiX = new double[field.GetUpperBound(0) + 1, field.GetUpperBound(1) + 1];
            var FiY = new double[field.GetUpperBound(0) + 1, field.GetUpperBound(1) + 1];
            for (int x = 0; x <= field.GetUpperBound(0); x++)
            {
                for (int y = 0; y <= field.GetUpperBound(1); y++)
                {
                    FiX[x, y] = Math.Cos(2 * field[x, y]);
                    FiY[x, y] = Math.Sin(2 * field[x, y]);
                }
            }
            var FiXsmoothed = new double[field.GetUpperBound(0) + 1, field.GetUpperBound(1) + 1];
            var FiYsmoothed = new double[field.GetUpperBound(0) + 1, field.GetUpperBound(1) + 1];
            var fieldSmoothed = new double[field.GetUpperBound(0) + 1, field.GetUpperBound(1) + 1];
            for (int y = 0; y <= field.GetUpperBound(1); y++)
            {
                for (int x = 0; x <= field.GetUpperBound(0); x++)
                {

                    double resultX = 0, resultY = 0;
                    int count = 0;
                    for (int i = -1; i < 2; i++)
                    {
                        if (x + i < 0 || x + i > field.GetUpperBound(0)) continue;
                        for (int j = -1; j < 2; j++)
                        {
                            if (y + j < 0 || y + j > field.GetUpperBound(1)) continue;
                            resultX += FiX[x + i, y + j];
                            resultY += FiY[x + i, y + j];
                            count++;
                        }
                    }
                    FiXsmoothed[x, y] = resultX / count;
                    FiYsmoothed[x, y] = resultY / count;
                    // бага тут
                    // = 0.5*Math.Atan(FiYsmoothed[x, y]/FiXsmoothed[x, y]);
                    var xx = FiXsmoothed[x, y];
                    var yy = FiYsmoothed[x, y];
                    if (xx > 0 && yy >= 0)
                        fieldSmoothed[x, y] = Math.Atan(yy / xx) / 2;
                    if (xx > 0 && yy < 0)
                        fieldSmoothed[x, y] = Math.Atan(yy / xx) / 2 + Math.PI;
                    if (xx < 0)
                        fieldSmoothed[x, y] = Math.Atan(yy / xx) / 2 + Math.PI / 2;
                    if (xx == 0 && yy > 0)
                        fieldSmoothed[x, y] = Math.PI / 4;
                    if (xx == 0 && yy < 0)
                        fieldSmoothed[x, y] = 3 * Math.PI / 4;
                    if (double.IsNaN(xx) || double.IsNaN(yy)) fieldSmoothed[x, y] = double.NaN;
                }
            }
            return fieldSmoothed;
        }

        public static void SaveField(double[,] field, string name)
        {
            var bmp = new Bitmap(5 * field.GetUpperBound(0), 5 * field.GetUpperBound(1));
            var gfx2 = Graphics.FromImage(bmp);
            gfx2.FillRectangle(Brushes.White, 0, 0, 5 * field.GetUpperBound(0), 5 * field.GetUpperBound(1));

            for (int x = 0; x < field.GetUpperBound(0); x++)
            {
                for (int y = 0; y < field.GetUpperBound(1); y++)
                {
                    if (double.IsNaN(field[x, y])) continue;
                    var k = Math.Tan(field[x, y]);

                    if (Math.Abs(k) > 1)
                    {
                        //берём по горизонтальной границе
                        float pointX = 2.0f / (float)k;
                        gfx2.DrawLine(Pens.Black, new PointF(-pointX + 5.0f * x, -2.0f + 5.0f * y),
                                      new PointF(pointX + 5.0f * x, 2.0f + 5.0f * y));
                    }
                    else
                    {
                        float pointY = (float)k * 2.0f;
                        gfx2.DrawLine(Pens.Black, new PointF(2.0f + 5.0f * x, pointY + 5.0f * y),
                                      new PointF(-2.0f + 5.0f * x, -pointY + 5.0f * y));
                    }
                }
            }

            gfx2.Save();

            bmp.Save(name, ImageFormat.Bmp);
        }
    }
}
