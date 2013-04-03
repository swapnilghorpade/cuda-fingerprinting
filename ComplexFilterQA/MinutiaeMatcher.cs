using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;

namespace ComplexFilterQA
{
    public struct Minutia
    {
        public int X;
        public int Y;
        public double Angle;

        public static bool operator ==(Minutia m1, Minutia m2)
        {
            return m1.Angle == m2.Angle && m1.X == m2.X && m1.Y == m2.Y;
        }

        public static bool operator !=(Minutia m1, Minutia m2)
        {
            return !(m1 == m2);
        }
    }

    class MinutiaeMatcher
    {
        private const double DistanceToleranceBox = 3;
        private const double MatchingToleranceBox = 6;
        private const double AngleToleranceBox = Math.PI/8;

        public static void SaveMinutiae(List<Minutia> lst, string filename)
        {
            using (FileStream fs = new FileStream(filename, FileMode.Create,FileAccess.Write))
            {
                using (BinaryWriter sr = new BinaryWriter(fs))
                {
                    sr.Write(lst.Count);
                    foreach(var m in lst)
                    {
                        sr.Write(m.X);
                        sr.Write(m.Y);
                        //sr.Write(m.Angle);
                    }
                }
            }
        }

        public static List<Minutia> LoadMinutiae(string filename)
        {
            var result = new List<Minutia>();
            using (FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read))
            {
                using (BinaryReader sr = new BinaryReader(fs))
                {
                    int i = sr.ReadInt32();
                    while(i-->0)
                    {
                        Minutia m = new Minutia();
                        m.X = sr.ReadInt32();
                        m.Y = sr.ReadInt32();
                        //m.Angle = sr.ReadDouble();
                        result.Add(m);
                    }
                }

            }
            return result;
        }

        //internal static List<Minutia> TranslateToSecond(List<Minutia> minutiae1, Minutia referenceMinutia1, Minutia referenceMinutia2, double rotationAngle)
        //{
        //    //from m1 to m2. I DEMAND IT
        //    var m1 = new List<Minutia>();

        //    var cos = Math.Cos(rotationAngle);
        //    var sin = -Math.Sin(rotationAngle);
        //    foreach (var m in minutiae1)
        //    {
        //        double xDash = m.X - referenceMinutia1.X;
        //        double yDash = m.Y - referenceMinutia1.Y;
        //        xDash = cos * xDash - sin * yDash;
        //        yDash = sin * xDash + cos * yDash;

        //        m1.Add(new Minutia() { X = (int)Math.Round(xDash) + referenceMinutia2.X, Angle = m.Angle, Y = (int)Math.Round(yDash) + referenceMinutia2.Y });
        //    }

        //    return m1;
        //}

        internal static double TranslateAndMatch(List<Minutia> minutiae1, Minutia referenceMinutia1, List<Minutia> minutiae2, Minutia referenceMinutia2, double rotationAngle)
        {
            //from m1 to m2. I DEMAND IT
            var m1 = minutiae1.Select(x => new Minutia() { Angle = x.Angle, X = x.X - referenceMinutia1.X, Y = x.Y - referenceMinutia1.Y }).Reverse();
            var m2 = minutiae2.Select(x => new Minutia() { Angle = x.Angle, X = x.X - referenceMinutia2.X, Y = x.Y - referenceMinutia2.Y }).Reverse().ToList();

            var cos = Math.Cos(rotationAngle);
            var sin = -Math.Sin(rotationAngle);
            foreach (var m in m1)
            {
                var xDash = cos * m.X - sin * m.Y;
                var yDash = sin * m.X + cos * m.Y;

                if (m2.Any(x => (xDash - x.X) * (xDash - x.X) + (yDash - x.Y) * (yDash - x.Y) < MatchingToleranceBox * MatchingToleranceBox))
                {
                    m2.Remove(m2.OrderBy(x => (xDash - x.X) * (xDash - x.X) + (yDash - x.Y) * (yDash - x.Y)).First());
                }
            }
            var total = (minutiae2.Count - m2.Count) / ((double)(minutiae2.Count+minutiae1.Count)/2);
            return total;
        }

        //internal static List<Point> GetBestFoundMinutiaCorrelation(List<Minutia> minutiae1, List<Minutia> minutiae2)
        //{
        //    Stopwatch sw = new Stopwatch();
        //    sw.Start();
        //    // calculate square distances
        //    var ds1 = CalculateDistances(minutiae1);
        //    var ds2 = CalculateDistances(minutiae2);

        //    var matchedVertices = new Dictionary<Point, List<Point>>();

        //    foreach (var tuple1 in ds1)
        //    {
        //        foreach (var tuple2 in ds2)
        //        {
        //            if (tuple1.Item2 - DistanceToleranceBox > tuple2.Item2) continue;
        //            if (tuple1.Item2 + DistanceToleranceBox < tuple2.Item2) break;

        //            var key = new Point(tuple1.Item1.X, tuple2.Item1.X);
        //            var value = new Point(tuple1.Item1.Y, tuple2.Item1.Y);
        //            if (!matchedVertices.ContainsKey(key)) matchedVertices[key] = new List<Point>();
        //            matchedVertices[key].Add(value);
        //            if (!matchedVertices.ContainsKey(value)) matchedVertices[value] = new List<Point>();
        //            matchedVertices[value].Add(key);

        //            key = new Point(tuple1.Item1.X, tuple2.Item1.Y);
        //            value = new Point(tuple1.Item1.Y, tuple2.Item1.X);
        //            if (!matchedVertices.ContainsKey(key)) matchedVertices[key] = new List<Point>();
        //            matchedVertices[key].Add(value);
        //            if (!matchedVertices.ContainsKey(value)) matchedVertices[value] = new List<Point>();
        //            matchedVertices[value].Add(key);
        //        }
        //    }

        //    var maxReliability = 0;
        //    var pt1 = new Point(0, 0);
        //    List<Point> correlations = null;
        //    foreach (var matchedVertex in matchedVertices)
        //    {
        //        var vertices = (from point in matchedVertex.Value
        //                        let intersection = matchedVertices[point].Intersect(matchedVertex.Value)
        //                        let nonAmbiguous = intersection.GroupBy(x => x.Y).Where(x => x.Count() == 1).SelectMany(x => x.Take(1)).
        //                        GroupBy(x => x.X).Where(x => x.Count() == 1).SelectMany(x => x.Take(1)).ToList()
        //                        where nonAmbiguous.Any()
        //                        orderby nonAmbiguous.Count() descending
        //                        select new { Point = point, Intersection = nonAmbiguous, Reliability = nonAmbiguous.Count() }).ToList();
        //        if (vertices.Any())
        //        {
        //            var measure = vertices.First().Reliability;
        //            if (measure > maxReliability)
        //            {
        //                maxReliability = measure;
        //                correlations =
        //                    new List<Point>(vertices.First().Intersection)
        //                        {
        //                            vertices.First().Point,
        //                        };
        //                correlations.Insert(0, matchedVertex.Key);
        //            }
        //        }

        //    }
        //    sw.Stop();
        //    return correlations;
        //}

        //public static double CalculateRotation(List<Minutia> minutiaeFrom, List<Minutia> minutiaeTo, Point centerPoint, List<Point> correlation)
        //{
        //    var centerFrom = minutiaeFrom[centerPoint.X];
        //    var centerTo = minutiaeTo[centerPoint.Y];

        //    double result = 0;

        //    foreach (var point in correlation)
        //    {
        //        var pointFrom = minutiaeFrom[point.X];
        //        var pointTo = minutiaeTo[point.Y];

        //        var angleFrom = DetermineAngle(centerFrom, pointFrom);
        //        var angleTo = DetermineAngle(centerTo, pointTo);

        //        result += angleTo - angleFrom;
        //    }

        //    return result/correlation.Count;
        //}

        private static double DetermineAngle(Minutia begin, Minutia end)
        {
            var dx = end.X - begin.X;
            var dy = begin.Y-end.Y;
            
            return Math.Atan2(dy, dx);
        }

        //point is used for minutiae index
        //private static List<Tuple<Point,double>> CalculateDistances(List<Minutia> minutiae1)
        //{
        //    var result = new List<Tuple<Point, double>>();
        //    for(int i=0;i<minutiae1.Count;i++)
        //    {
        //        var m1 = minutiae1[i];
        //        for (int j = i+1; j < minutiae1.Count; j++)
        //        {
        //            var m2 = minutiae1[j];
        //            var d = DetermineDistance(m1, m2);
        //            result.Add(Tuple.Create(new Point(i, j), d));
        //        }
        //    }
        //    return result.OrderBy(x=>x.Item2).ToList();
        //}

        public static double Match(List<Minutia> minutiae1, List<Minutia> minutiae2)
        {
            var sw = new Stopwatch();
            sw.Start();
            double max = 0;
            int count = 0;
            foreach (var core1 in minutiae1)
            {
                foreach (var to1 in minutiae1)
                {
                    if (core1 == to1) continue;
                    var angle1 = DetermineAngle(core1, to1);
                    var length1 = DetermineDistance(core1, to1);

                    foreach (var core2 in minutiae2)
                    {
                        var others2 = minutiae2.Except(new List<Minutia>() {core2});
                        foreach (var to2 in others2)
                        {
                            if (core2 == to2) continue;
                            var angle2 = DetermineAngle(core2, to2);
                            if (Math.Abs(angle1 - angle2) > AngleToleranceBox) 
                                continue;
                            var length2 = DetermineDistance(core2, to2);
                            if (Math.Abs(length1 - length2) > DistanceToleranceBox) continue;
                            count++;
                            var score = TranslateAndMatch(minutiae1, core1, minutiae2, core2, angle2 - angle1);
                            if (score > max) max = score;
                        }
                    }
                }
            }
            //var correlation = GetBestFoundMinutiaCorrelation(minutiae1, minutiae2);
            //var rotation = CalculateRotation(minutiae1, minutiae2, correlation.First(), correlation.Skip(1).ToList());

            //var anotherScore = TranslateAndMatch(minutiae1, minutiae1[correlation.First().X], minutiae2, minutiae2[correlation.First().Y], rotation);
            sw.Stop();
            return max;
        }

        private static double DetermineDistance(Minutia m1, Minutia m2)
        {
            return Math.Sqrt((m1.X - m2.X) * (m1.X - m2.X) + (m1.Y - m2.Y) * (m1.Y - m2.Y));
        }
    }
}
