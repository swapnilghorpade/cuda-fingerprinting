using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC
{
    public static class Numeration
    {
        public static int[] numerizationBlock(int[,,] cylinder, int xNs, int yNs, int zNd, double r) { //
          
            float sizeOneCube = 2*(float)r/xNs;

            List<int> vector = new List<int>();

            for (int z = 0; z < zNd; z++)
            {
                for (int y = 0; y < yNs; y++)
                {
                    for (int x = 0; x < xNs; x++)
                    {
                        double d = Math.Sqrt(Math.Pow(((x+0.5)*sizeOneCube - r),2)
                            + Math.Pow(((y + 0.5) * sizeOneCube - r),2));
                        if (d <= r) {
                            vector.Add(cylinder[x,y,z]);
                        }
                    }
                }
            }

            int sizeResVec = (vector.Count + 31)/32;

            int [] resVec = new int[sizeResVec];

            for (int i = 0; i < sizeResVec; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    if (j + i*32 < vector.Count)
                    {
                        resVec[i] = resVec[i] | (vector[j + i*32] << (31 - j));
                    }
                }
            }
            return resVec;
        }
    }
}