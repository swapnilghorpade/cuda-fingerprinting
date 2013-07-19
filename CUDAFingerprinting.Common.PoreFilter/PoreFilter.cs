using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Common.PoreFilter
{
    public class PoreFilter
    {
        public static void DeletePores(double[,] field)
        {
            for (int i = 1; i < field.GetLength(0)-1; i++)
                for (int j = 1; j < field.GetLength(1)-1; j++)
                    if (field[i, j] == 255)
                    {
                        int count = 0;
                        for (int u = -1; u < 2; u++)
                            for (int v = -1; v < 2; v++)
                                if (field[u + i, v + j] == 255)
                                    count++;
                        if (count <= 3)
                            field[i, j] = 0;
                    }
            for (int i = 1; i < field.GetLength(0) - 1; i++)
                for (int j = 1; j < field.GetLength(1) - 1; j++)
                    if (field[i, j] == 0)
                    {
                        int count = 0;
                        for (int u = -1; u < 2; u++)
                            for (int v = -1; v < 2; v++)
                                if (field[u + i, v + j] == 0)
                                    count++;
                        if (count <= 3)
                            field[i, j] = 255;
                    }
        }
    }
}
