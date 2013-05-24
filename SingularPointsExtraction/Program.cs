using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;

namespace SingularPointsExtraction
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = "D:/img/1.tif";
            double[,] img = ImageHelper.LoadImage(path);
            img = ImageEnhancementHelper.EnhanceImage(img);
            SPByComplexFiltering.ExtractSP(img);
            SPByPoincareIndex.ExtractSP(img);
        }
    }
}
