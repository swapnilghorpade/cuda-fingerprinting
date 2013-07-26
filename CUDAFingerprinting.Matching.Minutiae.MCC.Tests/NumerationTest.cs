using System;
using System.Collections;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.TemplateBuilding.Minutiae.MCC;

namespace CUDAFingerprinting.Matching.Minutiae.MCC.Tests
{
    [TestClass]
    public class NumerationTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            int Ns = 16;
            int Nd = 6;
            int R = 70;
            int[,,] m = new int[Ns,Ns,Nd];
            for (int i = 0; i < Nd; i++)
            {
                for (int j = 0; j < Ns; j++)
                {
                    for (int k = 0; k < Ns; k++)
                    {
                        //m[k, j, i] = 1;// (k % 2 + j % 2 + i % 2) % 2;
                        if (k <= j)
                        {
                            m[k, j, i] = 1;
                        }
                    }
                }
            }
            System.Console.WriteLine(Numeration.numerizationBlock(m));
            
        }

        [TestMethod]
        public void TestMethod2()
        {
            int Ns = 16;
            int Nd = 6;
            int R = 70;
            int[, ,] m = new int[Ns, Ns, Nd];
            for (int i = 0; i < Nd; i++)
            {
                for (int j = 0; j < Ns; j++)
                {
                    for (int k = 0; k < Ns; k++)
                    {
                        if (k <= j)
                        {
                            m[k, j, i] = 1;
                        }
                    }
                }
            }

            BitArray b = new BitArray(Numeration.fullNumerationBlock(m));
            bool[] bits = new bool[b.Count];
            b.CopyTo(bits, 0);
            byte[] bitValues = bits.Select(bit => (byte)(bit ? 1 : 0)).ToArray();

            //System.Console.WriteLine(Numeration.fullNumerationBlock(m));
            System.Console.WriteLine("numBit");
            for (int i = 0; i < bitValues.GetLength(0); i++)
            {
                if (i%16 == 0)
                {
                    System.Console.WriteLine();
                }
                if (i%(16*16) == 0)
                {
                    System.Console.WriteLine();
                }
                System.Console.Write(bitValues[i]);
            }
        }
    }
}
