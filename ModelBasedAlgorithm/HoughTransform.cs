using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelBasedAlgorithm
{
    internal struct ParameterStruct
    {
        public int X;
        public int Y;
        public int P;
        public double Tetta;
        public int Vote;
    }

    // find core point... yet...

    internal class HoughTransform
    {
        private List<Tuple<int, int>> singularPointsPI = new List<Tuple<int, int>>();
        private List<ParameterStruct> parameterSpase = new List<ParameterStruct>();
        private double[,] featureSpace;
        private double backgroundOrientation = 0;
        private Tuple<int, int> point;

        public HoughTransform(List<Point> singularPointsPI)
        {
            this.singularPointsPI = singularPointsPI;
        }

        public void Transform(Tuple<int, int> point, double[,] featureSpace, double backgroundOrientation)
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
                    for (int paramIndex = 0; paramIndex < parameterSpase.Count; paramIndex++)
                    {
                        if (WhetherToVote(i, j, featureSpace[i, j], parameterSpase[paramIndex])
                        {
                            parameterSpase[paramIndex].Vote++;
                        }
                    }
                }
            }
        }

        private bool WhetherToVote(int x, int y, int value, ParameterStruct core)
        {
            return Math.Tan(2 * (value - backgroundOrientation)) * (x - core.X) == y - core.Y;
        }

        private void Initialize()
        {
            foreach (Tuple<int, int> singularPoint in singularPointsPI)
            {
                parameterSpase.Add(CalculateParameter(singularPoint));
            }
        }

        private ParameterStruct CalculateParameter(Tuple<int, int> singularPoint)
        {
            double k = Math.Tan(2 * (featureSpace[singularPoint.Item1, singularPoint.Item2] - backgroundOrientation));
            double tetta = Math.PI / 2 + Math.Atan(k);
            int p = (singularPoint.Y - k) / Math.Sqrt(k * k + 1);

            return new ParameterStruct() { X = singularPoint.Item1, Y = singularPoint.Item2, P = p, Tetta = tetta, Vote = 0 };
        }

        internal List<Tuple<int, int>> FilterThreshold()
        {
            int threshold = (int)(Constants.W * Constants.W / 2);
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();

            parameterSpase = parameterSpase.FindAll(parameter => parameter.Vote > threshold);

            foreach (ParameterStruct param in parameterSpase)
            {
                result.Add(new Tuple(param.X, param.Y));
            }

            return result;
        }
    }
}
