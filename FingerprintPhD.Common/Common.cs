using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;

namespace FingerprintPhD.Common
{
    public class Common
    {
        const string  Folder = "C:\\temp\\";

        public static int[] Convert2DArrayTo1D(int[,] arr)
        {
            var width = arr.GetLength(0);
            var height = arr.GetLength(1);
            var res = new int[width*height];
            for (int y = 0; y < width; y++)
            {
                for (int x = 0; x < height; x++)
                {
                    res[y*width + x] = arr[x, y];
                }
            }
            return res;
        }

        public static int[,] Convert1DArrayTo2D(int[] arr, int lineWidth)
        {
            var height = arr.Length/lineWidth;
            var res = new int[lineWidth , height];
            for (int i = 0; i < arr.Length;i++ )
            {
                var x = i%lineWidth;
                var y = i/lineWidth;
                res[x, y] = arr[i];
            }
            return res;
        }

        public static void SaveAndShowImage(int[,] imgBytes, string additionalName = "")
        {
            var width = imgBytes.GetLength(0);
            var height = imgBytes.GetLength(1);
            var img = new Bitmap(width, height);
            for(int x=0;x<width;x++)
            {
                for(int y=0;y<height;y++)
                {
                    var color = imgBytes[x, y];
                    img.SetPixel(x,y,Color.FromArgb(color,color,color));
                }
            }

            var filename = Folder +
                           (string.IsNullOrEmpty(additionalName) ? string.Empty : string.Format("_{0}_", additionalName)) +
                           Guid.NewGuid() + ".png";

            img.Save(filename,ImageFormat.Png);

            Process.Start(filename);
        }

        public static void SaveFingerCode(List<double> fCode, int nBands, int nSectors, int nFilters, int bandRadius,
                                           int holeRadius)
        {
            int center = nBands*bandRadius + holeRadius;
            var size = 2*(center)+1;

            int[,] imgBytes = new int[size*nFilters,size];

            for (int n = 0; n < nFilters;n++ )

                for (int x = 0; x < size; x++)
                {
                    for (int y = 0; y < size; y++)
                    {
                        var height = center - y;
                        var length = x - center;

                        var color = 255;

                        var distance = Math.Sqrt(height * height + length * length);

                        if (distance >= holeRadius && distance <= holeRadius + nBands * bandRadius)
                        {
                            var bandNumber = (int)((distance - holeRadius) / bandRadius);

                            if (bandNumber == nBands) bandNumber--;

                            var angle = 0.0;

                            if (length != 0)
                            {
                                angle = Math.Atan((double)height / length);
                            }
                            else angle=Math.PI/2*((height < 0) ? -1 : 1);

                            if (length < 0)
                            {
                                angle += Math.PI;
                            }
                            if (angle < 0) angle += Math.PI*2;
                            var sectorNumber = (int)(angle / (Math.PI * 2) * nSectors);

                            color = (int)(2.0 * fCode[n*nBands*nSectors+sectorNumber + bandNumber * nSectors]); //as there are 0-128
                        }
                        imgBytes[n*size+x, y] = color;
                    }
                }

            SaveAndShowImage(imgBytes);
        }
    }
}
