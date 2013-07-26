using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CUDAFingerprinting.TemplateBuilding.Minutiae.MCC;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC
{
    public static class Numeration
    {
        public static int[] numerizationBlock(int[,,] cylinder) { //

            int xNs = Constants.Ns;
            int yNs = Constants.Ns;
            int zNd = Constants.Nd;
            double r = Constants.R;

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
   //                         System.Console.Write(cylinder[x, y, z]);
                    }
   //                 System.Console.WriteLine();
                }
   //             System.Console.WriteLine();
            }

            int sizeResVec = (vector.Count + 31)/32;

            int [] resVec = new int[sizeResVec];

            for (int i = 0; i < sizeResVec; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    if (j + i*32 < vector.Count)
                    {
                        resVec[i] = resVec[i] | (vector[j + i*32] << j);
                    }
                }
                BitArray b = new BitArray(new int[] { resVec[i] });
                bool[] bits = new bool[b.Count];
                b.CopyTo(bits, 0);
            }
            return resVec;
        }
    
        public static int[] fullNumerationBlock(int[,,] cylinder)
        {
            int xNs = Constants.Ns;
            int yNs = Constants.Ns;
            int zNd = Constants.Nd;
            double r = Constants.R;

            float sizeOneCube = 2*(float)r/xNs;

            List<int> vector = new List<int>();

            for (int z = 0; z < zNd; z++)
            {
                for (int y = 0; y < yNs; y++)
                {
                    for (int x = 0; x < xNs; x++)
                    {
                            vector.Add(cylinder[x,y,z]);
                            System.Console.Write(cylinder[x, y, z]);
                    }
                    System.Console.WriteLine();
                }
                System.Console.WriteLine();
            }

            int sizeResVec = (vector.Count + 31)/32;

            int [] resVec = new int[sizeResVec];

            for (int i = 0; i < sizeResVec; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    if (j + i*32 < vector.Count)
                    {
                        resVec[i] = resVec[i] | (vector[j + i*32] << (j));
                    }
                }
                BitArray b = new BitArray(new int[] { resVec[i] });
                bool[] bits = new bool[b.Count];
                b.CopyTo(bits, 0);
            }
            return resVec;
        }

    }
}