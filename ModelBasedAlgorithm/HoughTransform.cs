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
        public double P;
        public double Tetta;
        public int Vote;
    }

    // find core point... yet...

    internal class HoughTransform
    {
        private List<Tuple<int, int>> singularPointsPI = new List<Tuple<int, int>>();
        private List<ParameterStruct> parameterSpase = new List<ParameterStruct>();
        private double[,] orientationField;
        private FeatureSpaceStruct featureSpace;
        private double backgroundOrientation = 0;
        private Tuple<int, int> point;
        public int max = 0;

        public HoughTransform(List<Tuple<int, int>> singularPointsPI, double[,] orientationField)
        {
            this.singularPointsPI = singularPointsPI;
            this.orientationField = orientationField;
        }

        public void Transform(Tuple<int, int> point, FeatureSpaceStruct featureSpace, double backgroundOrientation)
        {
            this.point = point;
            this.featureSpace = featureSpace;
            this.backgroundOrientation = backgroundOrientation;

            Initialize();

            int xLength = featureSpace.FeatureSpace.GetLength(0);
            int yLength = featureSpace.FeatureSpace.GetLength(1);

            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    for (int paramIndex = 0; paramIndex < parameterSpase.Count; paramIndex++)
                    {
                        if (WhetherToVote(i, j, featureSpace, parameterSpase[paramIndex]))
                        {
                            parameterSpase[paramIndex] = new ParameterStruct()
                            {
                                X = parameterSpase[paramIndex].X,
                                Y = parameterSpase[paramIndex].Y,
                                P = parameterSpase[paramIndex].P,
                                Tetta = parameterSpase[paramIndex].Tetta,
                                Vote = parameterSpase[paramIndex].Vote + 1
                            };
                        }
                    }
                }
            }
        }

        private bool WhetherToVote(int x, int y, FeatureSpaceStruct featureSpace, ParameterStruct core)
        {
            return Math.Tan(2 * (featureSpace.FeatureSpace[x, y] - backgroundOrientation)) * (x + featureSpace.ShiftX - core.X) 
                == y + featureSpace.ShiftY - core.Y;
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
            //
            double k = Math.Tan(2 * (orientationField[singularPoint.Item1, singularPoint.Item2] - backgroundOrientation));
            double tetta = Math.PI / 2 + Math.Atan(k);
            double p = (singularPoint.Item2 - k) / Math.Sqrt(k * k + 1);

            if (p < 0)
            {
                p = -1 * p;
            }

            return new ParameterStruct() { X = singularPoint.Item1, Y = singularPoint.Item2, P = p, Tetta = tetta, Vote = 0 };
        }

        internal List<Tuple<int, int>> FilterThreshold()
        {
            int threshold = (int)(Constants.W * Constants.W / 2);
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();

           // parameterSpase = parameterSpase.FindAll(parameter => parameter.Vote > threshold);

            foreach (ParameterStruct param in parameterSpase)
            {
                if (max < param.Vote)
                {
                    max = param.Vote;
                }

                result.Add(new Tuple<int, int>(param.X, param.Y));
            }

            // max = 18

            return result;
        }
    }
}
