using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
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

            Complex[,] response3 = Symmetry.EstimatePS(level0, 0.6d, 3.2d);
            double[,] forDelta3 = level0.Select2D((value, row, column) => (level0[row, level0.GetLength(1) - column - 1]));
            //ImageHelper.SaveArray(forDelta, "D:/1-4.bmp");
            Complex[,] responseForDelta3 = Symmetry.EstimatePS(forDelta3, 0.6d, 3.2d);

            for (int i = 0; i < response3.GetLength(0); i++)
            {
                for (int j = 0; j < response3.GetLength(1); j++)
                {
                    double newMagnitude = response3[i, j].Magnitude * (1d - responseForDelta3[i, j].Magnitude);
                    response3[i, j] = Complex.FromPolarCoordinates(newMagnitude, response3[i,j].Phase);                    
                }
                
            }

            double[,] aaa = response3.Select2D((value, row, column) => (response3[row,column].Magnitude));
            ImageHelper.SaveArray(aaa, "D:/1-4.bmp");
            

            return null;
        }

    }
}
