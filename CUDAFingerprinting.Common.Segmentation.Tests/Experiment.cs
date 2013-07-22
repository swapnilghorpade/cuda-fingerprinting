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
                                                int windowSize, int[] mask, int maskWidth, int maskHight);

        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "PostProcessing")]
        private static extern void PostProcessing(int[] mask, int maskX, int maskY, int threshold);
        
        private double[,] img = ImageHelper.LoadImage(Resources._2_2);

        private List<int[,]> binaryImgList = new List<int[,]>
            {
                ImageHelper.LoadImageAsInt(Resources._2_2),
                ImageHelper.LoadImageAsInt(Resources._103_7),
                ImageHelper.LoadImageAsInt(Resources._65_8),
                ImageHelper.LoadImageAsInt(Resources._104_6),
                ImageHelper.LoadImageAsInt(Resources._103_4),
                ImageHelper.LoadImageAsInt(Resources._105_4)
            };
        private int windowSize = 12;
        private double weight = 0.3;
        private int threshold = 5;

        [TestMethod]
        public void ExperimentMethod()
        {
            double[,] img1 = ImageHelper.LoadImage(Resources._104_6);
            int[,] resultImg1;
            
            resultImg1 = Segmentator.Segmetator(img1, windowSize, weight, threshold);
            ImageHelper.SaveIntArray(resultImg1, Path.GetTempPath() + "Segm_104_6" + ".png");                       
        }

        [TestMethod]
        public void MakeBinFromImage()
        {
            int i = 9;
            //ImageHelper.SaveImageAsBinary("C:\\temp\\104_6.png",
            //    "C:\\temp\\104_6.bin");

            //ImageHelper.SaveImageAsBinary("C:\\MyOwnFingerprinting\\CUDAFingerprinting.Common.Segmentation.Tests\\Resources\\103_4.tif",
            //    "C:\\temp\\103_4.bin");
        }

        //[TestMethod]
        //public void GpuTest()
        //{
        //    string pathToSave = Path.GetTempPath() + "mask_";
        //    string pathToSave1 = Path.GetTempPath() + "maskP_";

        //    for (int k = 0; k < binaryImgList.Count; k++)
        //    {
        //        int maskX = (int)Math.Ceiling((double)binaryImgList[k].GetLength(0) / windowSize);
        //        int maskY = (int)Math.Ceiling((double)binaryImgList[k].GetLength(1) / windowSize);
        //        int[] mask = new int[maskX*maskY];
        //        float[] oneDimensionalBinaryImg = new float[binaryImgList[k].GetLength(0) * binaryImgList[k].GetLength(1)];

        //        for (int i = 0; i < binaryImgList[k].GetLength(0); i++)
        //        {
        //            for (int j = 0; j < binaryImgList[k].GetLength(1); j++)
        //            {
        //                oneDimensionalBinaryImg[j * binaryImgList[k].GetLength(0) + i] = binaryImgList[k][i, j];
        //            }
        //        }

        //        CUDASegmentator(oneDimensionalBinaryImg, binaryImgList[k].GetLength(0), binaryImgList[k].GetLength(1), 
        //                        (float)weight, windowSize, mask, maskX, maskY);
        //        SaveMask(mask, maskX, maskY, pathToSave + k + ".txt");

        //        PostProcessing(mask, maskX, maskY, threshold);
        //        SaveMask(mask, maskX, maskY, pathToSave1 + k + ".txt");
        //    }
        //}

        private void SaveMask(int[] mask, int width, int height, string path)
        {
            StreamWriter writer = new StreamWriter(path, false);

            for(int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    writer.Write(mask[i*width + j]);
                }

               writer.WriteLine(" ");
            }
           
            writer.Close();
        }
    }
}
