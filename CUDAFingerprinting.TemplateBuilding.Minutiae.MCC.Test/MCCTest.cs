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
        private int windowSize = 12;
        private double weight = 0.3;
        private int threshold = 5;
        private double board = 150;
        private double[,] img = ImageHelper.LoadImage(Resources._104_6);

        [TestMethod]
        public void TestMcc()
        {
            int[,] maskOfSegmentation2D = Segmentator.Segmetator(img, windowSize, weight, threshold);
            var thining = Thining.ThiningPicture(GlobalBinarization.Binarization(img, board));
            List<Minutia> minutiaList = MinutiaeDetection.FindMinutiae(thining);
            List<Minutia> validMinutiae = new List<Minutia>();

            foreach (Minutia minutia in minutiaList)
            {
                if (maskOfSegmentation2D[minutia.Y, minutia.X] == 1) // coordinates swap - ok
                {
                    validMinutiae.Add(minutia);
                }
            }

            var response = MCC.MCCMethod(validMinutiae, thining.GetLength(0), thining.GetLength(1));

            SaveResponse(response, validMinutiae);
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

            Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> response = MCC.MCCMethod(twoMinutiae, 364, 256);

            for (int i = 0; i < response.Count; i++)
            {
                Img3DHelper.Save3DAs2D(response[twoMinutiae[i]].Item1, Path.GetTempPath() + "valueN" + i);
                Img3DHelper.Save3DAs2D(response[twoMinutiae[i]].Item2, Path.GetTempPath() + "maskN" + i);
            }
        }

        [TestMethod]
        public void TestWithAdequateMinutiaeSet()
        {
            int[,] mask = Segmentator.Segmetator(img, windowSize, weight, threshold);
            double[,] binaryImage = img; //
            //---------------------------------------
            double sigma = 1.4d;
            double[,] smoothing = LocalBinarizationCanny.Smoothing(binaryImage, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            int sizeWin = 16;
            binaryImage = LocalBinarizationCanny.LocalBinarization(binaryImage, nonMax, sizeWin, 1.3d);
            //---------------------------------------
            binaryImage = Thining.ThiningPicture(binaryImage);
            //---------------------------------------
            List<Minutia> minutiae = MinutiaeDetection.FindMinutiae(binaryImage);
            for (int i = 0; i < minutiae.Count; i++)
            {
                if (mask[minutiae[i].Y, minutiae[i].X] == 0)
                {
                    minutiae.Remove(minutiae[i]);
                    i--;
                }
            }

            minutiae = MinutiaeDetection.FindBigMinutiae(minutiae);

            var response = MCC.MCCMethod(minutiae, img.GetLength(0), img.GetLength(1));

            SaveResponse(response, minutiae);
        }

        private void SaveResponse(Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> response, List<Minutia> minutiae)
        {
            for (int i = 0; i < response.Count; i++)
            {
                Img3DHelper.Save3DAs2D(response[minutiae[i]].Item1, Path.GetTempPath() + "valueN" + i);
                Img3DHelper.Save3DAs2D(response[minutiae[i]].Item2, Path.GetTempPath() + "maskN" + i);
            }
            //  Graphics graphics = Graphics.FromImage(Resources._104_6);
        }
    }
}
