using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

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
            var total = (minutiae2.Count - m2.Count) ;
            return total;
        }

        private static double DetermineAngle(Minutia begin, Minutia end)
        {
            var dx = end.X - begin.X;
            var dy = begin.Y-end.Y;
            
            return Math.Atan2(dy, dx);
        }

        public static double Match(List<Minutia> minutiae1, List<Minutia> minutiae2)
        {
            var tasks =
                new Tuple<Minutia, Minutia, Minutia, Minutia>[6000];

            var sw = new Stopwatch();
            sw.Start();
            double max = 0;
            int count = -1;
            Parallel.ForEach(minutiae1, new ParallelOptions() {MaxDegreeOfParallelism = 4},
                             core1 =>
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
                                                 int localCount = Interlocked.Increment(ref count);
                                                 tasks[localCount] = Tuple.Create(core1, to1, core2, to2);
                                             }
                                         }
                                     }
                                 });
            object _lock = new object();
            Parallel.ForEach(tasks.Take(count), new ParallelOptions() {MaxDegreeOfParallelism = 4},
                             x =>
                                 {
                                     var angle1 = DetermineAngle(x.Item1, x.Item2);
                                     var angle2 = DetermineAngle(x.Item3, x.Item4);
                                     var score = TranslateAndMatch(minutiae1, x.Item1, minutiae2, x.Item3, angle2 - angle1);
                                     lock(_lock)
                                     {
                                         if (score > max) max = score;
                                     }
                                 });

            sw.Stop();
            return max;
        }

        private static double DetermineDistance(Minutia m1, Minutia m2)
        {
            return Math.Sqrt((m1.X - m2.X) * (m1.X - m2.X) + (m1.Y - m2.Y) * (m1.Y - m2.Y));
        }
    }
}
