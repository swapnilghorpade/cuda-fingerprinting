using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelBasedAlgorithm
{
    internal struct ParameterStruct
    {
        public int P;
        public double Tetta;
        public int Vote;
    }

    internal class HoughTransform
    {
        private List<Point> singularPointsPI = new List<Point>();
        private List<ParameterStruct> parameterSpase = new List<ParameterStruct>();
        private double[,] featureSpace;
        private double backgroundOrientation = 0;
        private Point point = new Point();

        public HoughTransform(List<Point> singularPointsPI)
        {
            this.singularPointsPI = singularPointsPI;
        }

        public void Transform(Point point, double[,] featureSpace, double backgroundOrientation)
        {
            this.point = point;
            this.featureSpace = featureSpace;
            this.backgroundOrientation = backgroundOrientation;

            Initialize();

            int xLength = featureSpace.GetLength(0);
            int yLength = featureSpace.GetLength(1);

            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    if (is)
                }
            }
        }

        private void Initialize()
        {
            foreach (Point singularPoint in singularPointsPI)
            {
                parameterSpase.Add(CalculateParameter(singularPoint));
            }
        }

        private ParameterStruct CalculateParameter(Point singularPoint)
        {
            double k = Math.Tan(2 * (featureSpace[singularPoint.X, singularPoint.Y] - backgroundOrientation));
            double tetta = Math.PI / 2 + Math.Atan(k);
            int p = (singularPoint.Y - k) / Math.Sqrt(k * k + 1);

            return new ParameterStruct() { P = p, Tetta = tetta, Vote = 0 };
        }
    }
}
