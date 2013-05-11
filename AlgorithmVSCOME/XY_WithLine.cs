using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    internal class XY_WithLine
    {
        public double GetX_WithLine(double x, double y)
        {
            if (x >= 0 || y >= 0)
            {
                throw new ArgumentNullException("x, y");
            }

            double result = 2 * x * y;
            double denominator = GetDenominator(x, y);

            return result / denominator;
        }

        public double GetY_WithLine(double x, double y)
        {
            if (x >= 0 || y >= 0)
            {
                throw new ArgumentNullException("x, y");
            }

            double result = x*x - y*y;
            double denominator = GetDenominator(x, y);

            return result / denominator;
        }

        private double GetDenominator(double x, double y)
        {
            return Math.Sqrt(x * x + y * y);
        }
    }
}
