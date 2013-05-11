using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    public class Gaussian
    {
        public static double CalculateGaussian(double x, double y)
        {
            double result = -1 * (x*x + y*y);
            double denominator = 2 * Constants.Sigma * Constants.Sigma;

            result = Math.Exp(result / denominator);

            return result;
        }
    }
}
