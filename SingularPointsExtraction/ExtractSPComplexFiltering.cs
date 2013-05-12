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
            double[,] level1 = Size.Reduce2(level0, 2d);
            double[,] level2 = Size.Reduce2(level1, 2d);
            double[,] level3 = Size.Reduce2(level2, 2d);

            return null;
        }

    }
}
