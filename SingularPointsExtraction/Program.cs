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
            string path = "D:/1.tif";
            double[,] img = ImageHelper.LoadImage(path);
            SPByComplexFiltering.ExtractSP(img);
        }
    }
}
