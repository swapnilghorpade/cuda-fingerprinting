using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComplexFilterQA;

namespace SingularPointsExtraction
{
    class SPByComplexFiltering
    {
        static public Tuple<int, int> ExtractSP(double[,] img)
        {
            //make Gaussian pyramid
            double[,] level0 = img;
            double[,] level1 = ChangingSize.Reduce2(level0, 2d);
            double[,] level2 = ChangingSize.Reduce2(level1, 2d);
            double[,] level3 = ChangingSize.Reduce2(level2, 2d);

            ImageHelper.SaveArray(level1, "D:/1-1.bmp");
            ImageHelper.SaveArray(level2, "D:/1-2.bmp");
            ImageHelper.SaveArray(level3, "D:/1-3.bmp");

            return null;
        }

    }
}
