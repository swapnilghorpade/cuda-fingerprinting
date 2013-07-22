using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.Segmentation;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.MCC.Test
{
    [TestClass]
    public class MCCTest
    {

        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CUDASegmentator")]
        private static extern void CUDASegmentator(float[] img, int imgWidth, int imgHeight, float weightConstant,
                                                int windowSize, int[] mask, int maskWidth, int maskHight);

        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "PostProcessing")]
        private static extern void PostProcessing(int[] mask, int maskX, int maskY, int threshold);


        private int windowSize = 12;
        private double weight = 0.3;
        private int threshold = 5;

        [TestMethod]
        public void TestMcc()
        {
            Bitmap bitmap = Resources._104_6;
            var img = ImageHelper.LoadImage(bitmap);
            var binaryImg = ImageHelper.LoadImageAsInt(bitmap);

            int maskX = (int)Math.Ceiling((double)binaryImg.GetLength(0) / windowSize);
            int maskY = (int)Math.Ceiling((double)binaryImg.GetLength(1) / windowSize);

            int[] maskOfSegmentation = new int[maskX * maskY];

            float[] oneDimensionalBinaryImg = new float[binaryImg.GetLength(0) * binaryImg.GetLength(1)];

            for (int i = 0; i < binaryImg.GetLength(0); i++)
            {
                for (int j = 0; j < binaryImg.GetLength(1); j++)
                {
                    oneDimensionalBinaryImg[j * binaryImg.GetLength(0) + i] = binaryImg[i, j];
                }
            }

            //CUDASegmentator(oneDimensionalBinaryImg, binaryImg.GetLength(0), binaryImg.GetLength(1),
            //                  (float)weight, windowSize, maskOfSegmentation, maskX, maskY);
            //PostProcessing(maskOfSegmentation, maskX, maskY, threshold);

            bool[,] maskOfSegmentation2D = Segmentator.GetMask(maskOfSegmentation, maskY, img.GetLength(1), img.GetLength(0), windowSize);

            double board = 150;
            var thining = Thining.ThiningPicture(GlobalBinarization.Binarization(img, board));
            List<Minutia> minutiaList = MinutiaeDetection.FindMinutiae(thining);
            List<Minutia> validMinutiae = new List<Minutia>();
            foreach (Minutia minutia in minutiaList)
            {
                if (maskOfSegmentation2D[minutia.X, minutia.Y])
                {
                    validMinutiae.Add(minutia);
                }
            }

            MCC.MCCMethod(validMinutiae, thining.GetLength(1), thining.GetLength(0)); // is it right?

            //int[,,] value = MCC.Value;
            //int[,,] mask3d = MCC.Mask;
            //Img3DHelper.Save3DAs2D(value, Path.GetTempPath() + "value");
            //Img3DHelper.Save3DAs2D(mask3d, Path.GetTempPath() + "mask");
        }
        [TestMethod]
        public void SimpleTestMcc()
        {
            List<Minutia> twoMinutiae = new List<Minutia>();
            Minutia firstMinutia = new Minutia();
            firstMinutia.X = 40;
            firstMinutia.Y = 60;
            firstMinutia.Angle = 0;
            twoMinutiae.Add(firstMinutia);

            Minutia secondMinutia = new Minutia();
            secondMinutia.X = 70;
            secondMinutia.Y = 100;
            secondMinutia.Angle = Math.PI / 6;
            twoMinutiae.Add(secondMinutia);

            MCC.MCCMethod(twoMinutiae, 364, 256);

            Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> response = MCC.Response;
            for (int i = 0; i < response.Count; i++)
            {
                Img3DHelper.Save3DAs2D(response[twoMinutiae[i]].Item1, Path.GetTempPath() + "valueN" + i);
                Img3DHelper.Save3DAs2D(response[twoMinutiae[i]].Item2, Path.GetTempPath() + "maskN" + i);
            }

        }
    }
}
