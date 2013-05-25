using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelBasedAlgorithm
{
    internal class HoughTransform
    {
        private List<Tuple<int, int>> singularPointsPI = new List<Tuple<int, int>>();
        private List<Рarameter> parameterSpase = new List<Рarameter>();
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

        // Преобразование Хафа.
        // Есть поле пространство признаков (featureSpace) (поле направлений около особой точки)
        // и есть пространство параметров (parameterSpase) (пространство координат особых точек).
        // Направление фона (backgroundOrientation) определяется по одному из блоков около особой точки.
        //
        public void Transform(Tuple<int, int> point, FeatureSpaceStruct featureSpace, double backgroundOrientation)
        {
            this.point = point;
            this.featureSpace = featureSpace;
            this.backgroundOrientation = backgroundOrientation;

            InitializeParameterSpase();

            int xLength = featureSpace.FeatureSpace.GetLength(0);
            int yLength = featureSpace.FeatureSpace.GetLength(1);

            // Процедура голосования
            for (int i = 0; i < xLength; i++)
            {
                for (int j = 0; j < yLength; j++)
                {
                    for (int paramIndex = 0; paramIndex < parameterSpase.Count; paramIndex++)
                    {
                        // определяем будет ли точка из пространства признаков голосовать 
                        // за особую точку (точка пространства параметров)
                        if (WhetherToVote(i, j, featureSpace, parameterSpase[paramIndex]))
                        {
                            parameterSpase[paramIndex].IncreaseVote();
                        }
                    }
                }
            }
        }

        // Решение о голосовании принимается, если выполняется уравнение
        private bool WhetherToVote(int x, int y, FeatureSpaceStruct featureSpace, Рarameter core)
        {
            return Math.Tan(2 * (featureSpace.FeatureSpace[x, y] - backgroundOrientation)) * (x + featureSpace.ShiftX - core.X) 
                == y + featureSpace.ShiftY - core.Y;
        }

        private void InitializeParameterSpase()
        {
            foreach (Tuple<int, int> singularPoint in singularPointsPI)		
            {
                parameterSpase.Add(CalculateParameter(singularPoint));
            }
        }

        private Рarameter CalculateParameter(Tuple<int, int> singularPoint)
        {
            double k = Math.Tan(2 * (orientationField[singularPoint.Item1, singularPoint.Item2] - backgroundOrientation));
            double tetta = Math.PI / 2 + Math.Atan(k);
            double p = (singularPoint.Item2 - k) / Math.Sqrt(k * k + 1);

            if (p < 0)
            {
                p = -1 * p;
            }

            return new Рarameter(singularPoint.Item1, singularPoint.Item2, p, tetta);
        }

        // Отсекаем ложные особые точки.
        internal List<Tuple<int, int>> FilterThreshold()
        {
            int threshold = (int)(Constants.W * Constants.W / 2);
            List<Tuple<int, int>> result = new List<Tuple<int, int>>();

            // Отбрасываем точки с количеством голосов меньше половины голосующих
           // parameterSpase = parameterSpase.FindAll(parameter => parameter.Vote > threshold);

            foreach (Рarameter param in parameterSpase)
            {
                if (max < param.Vote)
                {
                    max = param.Vote;
                }

                result.Add(new Tuple<int, int>(param.X, param.Y));
            }

            return result;
        }
    }
}
