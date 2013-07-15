using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Segmentation.Tests
{
    [TestClass]
    public class Experiment
    {
        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CUDASegmentator")]
        private static extern void CUDASegmentator(float[] img, int imgWidth, int imgHeight, float weightConstant, 
                                                int windowSize, bool[] mask, int maskWidth, int maskHight);

        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "PostProcessing")]
        private static extern void PostProcessing(bool[] mask, int maskX, int maskY, int threshold);

        private double[,] img = ImageHelper.LoadImage(Resources._2_2);
        private int[,] binaryImg = ImageHelper.LoadImageAsInt(Resources._104_6);
        private int windowSize = 12;
        private double weight = 0.3;
        private int threshold = 5;

        [TestMethod]
        public void ExperimentMethod()
        {
            // double[,] img1 = ImageHelper.LoadImage(Resources._104_6);
            // double[,] img2 = ImageHelper.LoadImage(Resources._65_8);
            // double[,] img3 = ImageHelper.LoadImage(Resources._103_7);
            double[,] resultImg1;
            // double[,] resultImg2;
            // double[,] resultImg3;

            resultImg1 = Segmentator.Segmetator(img, windowSize, weight, threshold);
            ImageHelper.SaveArray(resultImg1,
                                  Path.GetTempPath() + "GOOD_IMAGE_2_2_resultImg_" + weight + "_" + threshold +
                                  ".png");

            //for (double weight = minValue; weight <= maxValue; weight+= 0.1)
            //{
            //    for (int currentThreshold = minThreshold; currentThreshold <= maxThreshold; currentThreshold++)
            //    {
            //        resultImg1 = Segmentator.Segmetator(img1, windowSize, weight, currentThreshold);
            //        ImageHelper.SaveArray(resultImg1, Path.GetTempPath() + "104_6_resultImg_" + weight + "_" + currentThreshold + ".png");
            //        resultImg2 = Segmentator.Segmetator(img2, windowSize, weight, currentThreshold);
            //        ImageHelper.SaveArray(resultImg2, Path.GetTempPath() + "65_8_resultImg_" + weight + "_" + currentThreshold + ".png");
            //        resultImg3 = Segmentator.Segmetator(img3, windowSize, weight, currentThreshold);
            //        ImageHelper.SaveArray(resultImg3, Path.GetTempPath() + "103_7_resultImg_" + weight + "_" + currentThreshold + ".png");
            //    }
            //}
        }

        [TestMethod]
        public void MakeBinFromImage()
        {
            ImageHelper.SaveImageAsBinary("C:\\Users\\Tanya\\Documents\\tests_data\\103_7.tif",
                "C:\\Users\\Tanya\\Documents\\tests_data\\103_7.bin");
        }

        [TestMethod]
        public void GpuTest()
        {
            string pathToSave = Path.GetTempPath() + "mask.txt";
            string pathToSave1 = Path.GetTempPath() + "mask1.txt";

            int maskX = CeilMod(img.GetLength(0), windowSize);
            int maskY = CeilMod(img.GetLength(1), windowSize);
            bool[] mask = new bool[maskX * maskY + 1];
            float[] oneDimensionalBinaryImg = new float[binaryImg.GetLength(0) * binaryImg.GetLength(1) + 1];

            for (int i = 0; i < binaryImg.GetLength(0); i++)
            {
                for (int j = 0; j < binaryImg.GetLength(1); j++)
                {
                    oneDimensionalBinaryImg[j * binaryImg.GetLength(0) + i] = binaryImg[i, j];
                }
            }
            mask[2] = true;
            //mask[] = true;
            CUDASegmentator(oneDimensionalBinaryImg, binaryImg.GetLength(0), binaryImg.GetLength(1), (float)weight, windowSize, mask, maskX, maskY);

            SaveMask(mask, maskX, maskY, pathToSave);
            
           // PostProcessing(mask, maskX, maskY, threshold);

           // SaveMask(mask, maskX, maskY, pathToSave1);
        }

        private int CeilMod(int x, int y)
        {
            return (x + y - 1) / y;
        }

        private void SaveMask(bool[] mask, int width, int height, string path)
        {
            StreamWriter writer = new StreamWriter(path, false);

            for(int i = 0; i < width; i++)
            {
                for(int j = 0; j < height; j++)
                {
                    writer.Write(mask[j*width + i] ? 1 : 0);
                }

               writer.WriteLine(" ");
            }
           
            writer.Close();
        }
    }
}
