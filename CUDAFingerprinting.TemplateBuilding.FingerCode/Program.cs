using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Threading.Tasks;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.TemplateBuilding.FingerCode
{
    internal class Program
    {
        private static void Main()
        {
            var img = new Bitmap(Image.FromFile(".\\Fingerprints\\102_5.tif"));
            var imgBytes = new int[img.Size.Width, img.Size.Height];
            for (var x = 0; x < img.Size.Width; x++)
                for (var y = 0; y < img.Size.Height; y++)
                {
                    var color = img.GetPixel(x, y);
                    imgBytes[x, y] = (int)(.299 * color.R + .587 * color.G + .114 * color.B);
                }

            var orfield = OrientationFieldGenerator.GenerateOrientationField(imgBytes);
            //VSCOMEdetector(orfield);
            int nFilters = 7;
            int nBands = 3;
            int bandRadius = 24;
            int holeRadius = 14;
            int nSectors = 9;


           
            var filterbank = new List<List<Tuple<int, int, double>>>();

            int[,] filtersInt = new int[32,nFilters*32];

            for (int i = 0; i < nFilters; i++)
            {
                filterbank.Add(CreateGaborFilter((1.0/10), Math.PI/nFilters*i));
            }

            // this code was used to test the similarity of the fingerprints
            //var min = filterbank.SelectMany(x=>x).Select(x=>x.Item3).Min();
            //var max = filterbank.SelectMany(x=>x).Select(x=>x.Item3).Max();

            //for (int i = 0; i < nFilters; i++)
            //{
            //    var filter = filterbank[i];
            //    foreach (var tuple in filter)
            //    {
            //        filtersInt[tuple.Item1, tuple.Item2 + i*32] = (int) ((tuple.Item3 - min)/(max - min)*255);
            //    }
            //}

            //Common.SaveAndShowImage(filtersInt);

            int size = nBands * nSectors*nFilters;

            int num = 60000;

            List<double>[] dbase = new List<double>[num];
            var rand = new Random();
            for (int w = 0; w < num; w++)
            {
                var lst = new List<double>();
                for(int s=0;s<size;s++)
                {
                    lst.Add(rand.NextDouble() * 128.0f);
                }
                dbase[w] = lst;
            }

            using (FileStream fs = new FileStream("C:\\temp\\fusrodah_cpu_results.csv", FileMode.Create))
            {
                using (StreamWriter streamWriter = new StreamWriter(fs))
                {
                    streamWriter.WriteLine("Find;Create;Match;Top");
                    for (int n = 0; n < 20; n++)
                    {
                        int radius = nBands*bandRadius + holeRadius + 16; //mask size

                        var sw = new Stopwatch();
                        sw.Start();
                        PerformNewDetectionAlgorithmTestfire(imgBytes);
                        streamWriter.Write(sw.ElapsedMilliseconds + ";");
                        sw.Stop();


                        sw.Restart();
                        var normalizedImage =
                            NormalizeImage(imgBytes, nBands, holeRadius, bandRadius,
                                           265, 292);
                        //Common.SaveAndShowImage(normalizedImage);
                        var filteredImages = new List<int[,]>();
                        for (int i = 0; i < nFilters; i++) filteredImages.Add(null);
                        Parallel.ForEach(filterbank, new ParallelOptions() {MaxDegreeOfParallelism = 4},
                                         filter =>
                                             {

                                                 var filtered = FilterWithGaborFilter(normalizedImage,
                                                                                      filter, nBands, holeRadius,
                                                                                      bandRadius,
                                                                                      265, 292);
                                                 //Common.SaveAndShowImage(filtered);
                                                 filteredImages[filterbank.IndexOf(filter)] = filtered;
                                             }

                            );
                        var fingercode = new List<double>();
                        for (int i = 0; i < nFilters*nBands*nSectors; i++)
                        {
                            fingercode.Add(0);
                        }
                        Parallel.ForEach(filteredImages,
                                         new ParallelOptions {MaxDegreeOfParallelism = 4},
                                         filteredImgBytes =>
                                             {
                                                 var fingercodePart = FormFingerCode(filteredImgBytes, nSectors, nBands,
                                                                                     holeRadius,
                                                                                     bandRadius);
                                                 int startIndex = filteredImages.IndexOf(filteredImgBytes)*
                                                                  nBands*nSectors;
                                                 foreach (var d in fingercodePart)
                                                 {
                                                     fingercode[startIndex++] = d;
                                                 }
                                             });
                        sw.Stop();
                        streamWriter.Write(sw.ElapsedMilliseconds + ";");
                        //Common.SaveFingerCode(fingercode, nBands, nSectors, nFilters, bandRadius, holeRadius);
                        sw.Restart();
                        Tuple<int, double>[] result = new Tuple<int, double>[num];

                        Parallel.ForEach(new[] {0, 1, 2, 3}, new ParallelOptions() {MaxDegreeOfParallelism = 4},
                                         (offset) =>
                                             {
                                                 for (int i = 0; i < num/4; i++)
                                                 {
                                                     var index = i*4 + offset;
                                                     result[index] = Tuple.Create(index,
                                                                                  CalculateDistance(fingercode,
                                                                                                    dbase[index]));
                                                 }
                                             });
                        sw.Stop();
                        streamWriter.Write(sw.ElapsedMilliseconds + ";");
                        sw.Restart();
                        result = result.OrderByDescending(x => x.Item2).ToArray();
                        sw.Stop();
                        streamWriter.WriteLine(sw.ElapsedMilliseconds);
                    }
                }
            }
        }

        #region VORIV

        private static double Gaussian(double x, double y, double sigma)
        {
            return Math.Exp(-(x*x + y*y)/2.0d/(sigma*sigma));
        }

        private static PointF VSCOMEdetector(double[,] orField, int[,] imgBytes)
        {
            var sigmaC = 1.5d;
            var regionSize = 7;
            var m = regionSize*regionSize;
            double max = 0;
            for (int X = regionSize / 2; X < orField.GetLength(0) - regionSize / 2;X++ )
            {
                for (int Y = regionSize / 2; Y < orField.GetLength(1) - regionSize / 2; Y++)
                {
                    //symmetry
                    double fy = 0d, fx = 0d;
                    for(int x = -regionSize/2;x<=regionSize/2;x++)
                    {
                        for(int y=-regionSize/2;y<0;y++)
                        {
                            //if(x<0&&y<0)
                            {
                                double sqrt = Math.Sqrt(x*x + y*y);
                                double xLine = 2d*x*y/sqrt;
                                double yLine = ((double) x*x - y*y)/sqrt;
                                fy+=Gaussian(x,y,sigmaC)*(yLine*Math.Cos(2*orField[x+X,y+Y])+xLine*Math.Sin(2*orField[x+X,y+Y]));
                                fx += Gaussian(x, y, sigmaC)*
                                      (xLine*Math.Cos(2*orField[x + X, y + Y]) - yLine*Math.Sin(2*orField[x + X, y + Y]));
                            }
                        }
                    }
                    double sym = Math.Sqrt(fy*fy + fx*fx)/m;
                    if(sym>max) max = sym;
                }
            }
            var bmp = new Bitmap(orField.GetLength(0) - regionSize+1, orField.GetLength(1) - regionSize+1);
            var imgBmp = new Bitmap(imgBytes.GetLength(0), imgBytes.GetLength(1));
            for (int X = regionSize / 2; X < orField.GetLength(0) - regionSize / 2; X++)
            {
                for (int Y = regionSize / 2; Y < orField.GetLength(1) - regionSize / 2; Y++)
                {
                    //symmetry
                    double fy = 0d, fx = 0d;
                    for (int x = -regionSize / 2; x <= regionSize / 2; x++)
                    {
                        for (int y = -regionSize / 2; y < 0; y++)
                        {
                            //if (x < 0 && y < 0)
                            {
                                double sqrt = Math.Sqrt(x * x + y * y);
                                double xLine = 2d * x * y / sqrt;
                                double yLine = ((double)x * x - y * y) / sqrt;
                                fy += Gaussian(x, y, sigmaC) * (yLine * Math.Cos(2 * orField[x + X, y + Y]) + xLine * Math.Sin(2 * orField[x + X, y + Y]));
                                fx += Gaussian(x, y, sigmaC) *
                                      (xLine * Math.Cos(2 * orField[x + X, y + Y]) - yLine * Math.Sin(2 * orField[x + X, y + Y]));
                            }
                        }
                    }
                    double sym = Math.Sqrt(fy * fy + fx * fx) / m;
                    if(!double.IsNaN(sym))
                    {
                        var intensity = (int)(sym / max * 255);
                        bmp.SetPixel(X - regionSize / 2, Y - regionSize / 2, Color.FromArgb(intensity, intensity, intensity));
                    }
                    
                }
            }
            bmp.Save("C:\\temp\\VSCOME.png", ImageFormat.Png);
            imgBmp.Save("C:\\temp\\VSCOME_on_Finger.png", ImageFormat.Png);
            return new PointF(0, 0);
        }

        #endregion

        #region someRogueAlgorithm
        private static void PerformNewDetectionAlgorithmTestfire(int[,] imgBytes)
        {
            int radius = 5;
                var squaredRadius = radius*radius;
                var orField =
                    OrientationFieldGenerator.GenerateOrientationField(imgBytes);
                //OrientationFieldGenerator.SaveField(orField,"C:\\temp\\testfire.png");

                var sumField =
                    new double[orField.GetUpperBound(0) + 1,
                        orField.GetUpperBound(1) + 1
                        ];
                Point maxCoords = new Point(0, 0);
                double max = 0;
                for (int x = 0; x <= orField.GetUpperBound(0); x++)
                {
                    for (int y = 0; y <= orField.GetUpperBound(1); y++)
                    {
                        for (int i = -radius; i <= radius; i++)
                        {
                            if (x + i < 0 || x + i > orField.GetUpperBound(0))
                                continue;
                            for (int j = -radius; j <= radius; j++)
                            {
                                if (y + j < 0 ||
                                    y + j > orField.GetUpperBound(1))
                                    continue;
                                if (j*j + i*i > squaredRadius ||
                                    j == 0 && i == 0)
                                    continue;

                                var pointAngle = orField[x + i, y + j];
                                if (double.IsNaN(pointAngle)) continue;
                                if (pointAngle > Math.PI*2)
                                    pointAngle -= Math.PI*2;
                                if (pointAngle < 0) pointAngle += Math.PI*2;

                                var baseAngle = Math.Atan2(j, i);
                                baseAngle -= Math.PI/2;
                                if (baseAngle < 0) baseAngle += Math.PI*2;
                                if (baseAngle > Math.PI*2)
                                    baseAngle -= Math.PI*2;

                                double diffAngle = pointAngle - baseAngle;

                                sumField[x, y] += Math.Abs(Math.Cos(diffAngle));

                            }
                        }

                        if (sumField[x, y] > max)
                        {
                            max = sumField[x, y];
                            maxCoords = new Point(x, y);
                        }
                    }
                }
                var imgCoords =
                    new PointF(
                        (0.5f + maxCoords.X)*(OrientationFieldGenerator.W - 1),
                        (0.5f + maxCoords.Y)*(OrientationFieldGenerator.W - 1));
        }
        #endregion

        #region Fingercode
        private static double CalculateDistance(IEnumerable<double> list, IList<double> list2)
        {
            return Math.Sqrt(list.Select((t, i) => t - list2[i]).Sum(r => r*r));
        }


        private static List<Tuple<int, int, double>> CreateGaborFilter(double frequency, double angle)
        {
            var filter = new List<Tuple<int, int, double>>();
            const int xCenter = 16;
            const int yCenter = 16;
            double deltaX = 1.0/frequency/2.3; // empirically determined parameter
            double deltaY = 1.0/frequency/2.3;
            for (int x = 0; x < 32; x++)
            {
                for (int y = 0; y < 32; y++)
                {
                    double xDash = Math.Sin(angle)*(x - xCenter) + Math.Cos(angle)*(y - yCenter);
                    double yDash = Math.Cos(angle)*(x - xCenter) - Math.Sin(angle)*(y - yCenter);
                    double exp = Math.Exp(-0.5*(xDash*xDash/deltaX/deltaX + yDash*yDash/deltaY/deltaY));
                    double cos = Math.Cos(2.0*Math.PI*frequency*xDash);
                    var value = exp*cos;
                    //if(Math.Abs(value)>=0.00)
                    filter.Add(new Tuple<int, int, double>(x, y, value));
                }
            }
            return filter;
        }

        private static int[,] FilterWithGaborFilter(int[,] imgBytes, List<Tuple<int, int, double>> filter, int numBands,
                                                    int holeRadius, int bandRadius, int referencePointX,
                                                    int referencePointY)
        {
            var upperBound = (holeRadius + numBands*bandRadius)*(holeRadius + numBands*bandRadius);
            var lowerBound = holeRadius*holeRadius;
            var size = 1 + 2*(holeRadius + numBands*bandRadius);
            var result = new int[size,size];
            for (int x = referencePointX - holeRadius - bandRadius*numBands;
                 x <= referencePointX + holeRadius + bandRadius*numBands;
                 x++)
            {
                for (int y = referencePointY - holeRadius - bandRadius*numBands;
                     y <= referencePointY + holeRadius + bandRadius*numBands;
                     y++)
                {
                    int X = x - referencePointX;
                    int Y = y - referencePointY;
                    var rad = X*X + Y*Y;
                    if (rad >= lowerBound && rad <= upperBound)
                    {
                        double color =
                            filter.Sum(tuple => tuple.Item3 * (double)imgBytes[x - 16 + tuple.Item1, y - 16 + tuple.Item2]);

                        // convolution
                        if (color < 0) color = 0;
                        if (color > 255) color = 255;
                        result[
                            x - referencePointX + holeRadius + bandRadius * numBands,
                            y - referencePointY + holeRadius + bandRadius * numBands] = (int)color;
                    }
                    else
                        result[x - referencePointX + holeRadius + bandRadius*numBands,
                               y - referencePointY + holeRadius + bandRadius*numBands] = 0;
                }
            }
            return result;
        }

        private const int BaseVariance = 100;
        private const int BaseMean = 100;

        private static int[,] NormalizeImage(int[,] imgBytes, int numBands, int holeRadius, int bandRadius, int centerX,
                                             int centerY)
        {
            List<int>[] buckets = new List<int>[2];
            buckets[0] = new List<int>();
            buckets[1] = new List<int>();
            double[] means = new double[2];
            double[] variances = new double[2];
            int[,] regionMap = new int[imgBytes.GetUpperBound(0) + 1,imgBytes.GetUpperBound(1) + 1];
            // dividing by regions
            for (int x = 0; x <= imgBytes.GetUpperBound(0); x++)
            {
                for (int y = 0; y <= imgBytes.GetUpperBound(1); y++)
                {
                    int X = x - centerX;
                    int Y = y - centerY;
                    double radius = Math.Sqrt(X*X + Y*Y);

                    buckets[radius >= holeRadius && radius < holeRadius + numBands*bandRadius ? 1 : 0].Add(
                        imgBytes[x, y]);
                    regionMap[x, y] = radius >= holeRadius + numBands*bandRadius ? 0 : 1;
                }
            }
            for (int i = 0; i < buckets.Count(); i++)
            {
                var values = buckets[i];
                var mean = means[i] = (double) values.Sum()/values.Count;
                variances[i] = (values.Select(v => (v - mean)*(v - mean)).Sum())/values.Count;
            }
            var result = new int[imgBytes.GetUpperBound(0) + 1,imgBytes.GetUpperBound(1) + 1];
            using (FileStream fs = new FileStream("C:\\temp\\nimage.bin", FileMode.Create))
            {
                var bw = new BinaryWriter(fs);
                for (int y = 0; y <= imgBytes.GetUpperBound(1); y++)
                {
                    for (int x = 0; x <= imgBytes.GetUpperBound(0); x++)
                    {
                        if (regionMap[x, y] == -1)
                            result[x, y] = imgBytes[x, y];
                        else
                        {
                            var mean = means[regionMap[x, y]];
                            var variance = variances[regionMap[x, y]];
                            var value = imgBytes[x, y];

                            result[x, y] = BaseMean;
                            if (variance > 0)
                            {
                                result[x, y] +=
                                    ((value > mean) ? (1) : (-1))*
                                    (int) Math.Sqrt(BaseVariance/variance*(value - mean)*(value - mean));

                                if (result[x, y] < 0) result[x, y] = 0;
                                if (result[x, y] > 255) result[x, y] = 255;
                                bw.Write(result[x, y]);
                            }
                        }
                    }
                }
                bw.Close();
                bw.Dispose();
            }
            return result;
        }

        private static IEnumerable<double> FormFingerCode(int[,] imgBytes, int numSectors, int numBands, int holeRadius,
                                                          int bandRadius)
        {
            Stopwatch sw = new Stopwatch();
            
            var sectorsMap = new List<int>[numBands*numSectors];
            var means = new double[numBands*numSectors];
            var variances = new double[numBands*numSectors];
            var bucketHash = new int[imgBytes.GetUpperBound(0) + 1,imgBytes.GetUpperBound(1) + 1];

            var center = (imgBytes.GetUpperBound(0) + 1)/2;
            for (int i = 0; i < numBands*numSectors; i++) sectorsMap[i] = new List<int>();
            // divide by sectors
            for (int x = 0; x <= imgBytes.GetUpperBound(0); x++)
            {
                for (int y = 0; y <= imgBytes.GetUpperBound(1); y++)
                {
                    int X = x - center;
                    int Y = y - center;
                    double radius = Math.Sqrt(X*X + Y*Y);
                    if (radius < holeRadius || radius >= holeRadius + numBands*bandRadius)
                    {
                        continue;
                    }
                    var bandBase = (int) ((radius - holeRadius)/bandRadius);

                    double angle = (X == 0) ? (Y > 0) ? (Math.PI/2) : (-Math.PI/2) : Math.Atan((double) Y/X);
                    if (X < 0) angle += Math.PI;
                    if (angle < 0) angle += 2*Math.PI;
                    var sectorNumber = (int) (angle/(Math.PI*2/numSectors));
                    sectorsMap[bandBase*numSectors + sectorNumber].Add(imgBytes[x, y]);
                    bucketHash[x, y] = bandBase*numSectors + sectorNumber;
                }
            }

            sw.Start();
            for (int i = 0; i < sectorsMap.Count(); i++)
            {
                var values = sectorsMap[i];
                var mean = means[i] = (double) values.Sum()/values.Count;
                variances[i] = (values.Select(v => Math.Abs(v - mean) ).Sum()) / values.Count;
            }
            var res = variances.ToList();
            sw.Stop();
            return res;
        }
        #endregion
    }
}
