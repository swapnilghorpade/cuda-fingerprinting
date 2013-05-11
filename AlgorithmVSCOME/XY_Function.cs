using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmVSCOME
{
    internal class XY_Function
    {
        private double u, v, sum_X, result_X, sum_Y, result_Y = 0;
        private int upperBound, lowerBound = 0;
        private XY_WithLine withLine = new XY_WithLine();

        public double X_Function
        {
            get { return result_X; }
        }

        public double Y_Function
        {
            get { return result_Y; }
        }

        public XY_Function(double u, double v)
        {
            this.u = u;
            this.v = v;

            upperBound = (int)(Constants.W / 2);
            lowerBound = -1 * upperBound;
        }

        internal void CalculateFunction()
        {
            double gaussian, xWithLine, yWithLine, arg, sin, cos = 0;

            for (int x = lowerBound; x <= upperBound; x++)
            {
                for (int y = lowerBound; y <= upperBound; y++)
                {
                    gaussian = Gaussian.CalculateGaussian(x, y);
                    xWithLine = withLine.GetX_WithLine(x, y);
                    yWithLine = withLine.GetY_WithLine(x, y);
                    arg = 2 * Tetta.GetTetta(u + y, v + x);
                    sin = Math.Sin(arg);
                    cos = Math.Cos(arg);

                    sum_X += gaussian * (xWithLine * cos - yWithLine * sin);
                    sum_Y += gaussian * (yWithLine * cos + xWithLine * sin);
                }

                result_X += sum_X;
                result_Y += sum_Y;
                sum_X = 0;
                sum_Y = 0;
            }
        }
    }
}
