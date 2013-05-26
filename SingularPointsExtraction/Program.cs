using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;
using System.IO;

namespace SingularPointsExtraction
{
    class Program
    {
        static void Main(string[] args)
        {
            //string path1 = "D:/img/101_4.tif";
            string[] pathes = Directory.GetFiles("D:/1/Db2_aEdited");
            string path;
            string path1;
            StreamWriter print1 = new StreamWriter("D:/1/outC1.txt", true);
            StreamWriter print2 = new StreamWriter("D:/1/outP1.txt", true);

            for (int i = 0; i < 10/*pathes.Length*/; i++)
            {
                path = pathes[i];
                Tuple<int, int> RedPoint = ImageHelper.FindRedPoint(path);
                path1 = path.Replace("D:/1/Db2_aEdited", "D:/1/Db2_a");
                double[,] img = ImageHelper.LoadImage(path);
                img = ImageEnhancementHelper.EnhanceImage(img);
                Tuple<int, int> point1 = SPByComplexFiltering.ExtractSP(img);
                Tuple<int, int> point2 = SPByPoincareIndex.ExtractSP(img);
                print1.WriteLine(GetDistance(RedPoint,point1));
                print2.WriteLine(GetDistance(RedPoint, point2));
            }
        }

        private static double GetDistance(Tuple<int, int> a, Tuple<int, int> b)
        {
            return Math.Sqrt( Math.Pow((a.Item1-b.Item1),2) + Math.Pow((a.Item2-b.Item2),2));
        }
    }
}
