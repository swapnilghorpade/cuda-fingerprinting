using System;
using System.Collections;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.Segmentation;
using CUDAFingerprinting.ImageEnhancement.LinearSymmetry;
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
            var thining = Thining.ThinPicture(GlobalBinarization.Binarization(img, board));
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

            SaveResponse(response, twoMinutiae);
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
            binaryImage = Thining.ThinPicture(binaryImage);
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

        [TestMethod]
        public void SimpleTestNumerationMcc()
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
                int[] valueN = Numeration.numerizationBlock(response[twoMinutiae[i]].Item1);
                int[] maskN = Numeration.numerizationBlock(response[twoMinutiae[i]].Item2);
                for (int j = 0; j < maskN.GetLength(0); j++)
                {
                    //System.Console.Write(valueN[j] + " ");
                    BitArray b = new BitArray(new int[] { maskN[j] });
                    bool[] bits = new bool[b.Count];
                    b.CopyTo(bits, 0);

                    if (j % (maskN.Count() / 6) == 0)
                    {
       //                 System.Console.Write(" j = " + j + "\n");
                    }

                    for (int k = 0; k < bits.GetLength(0); k++)
                    {
                        System.Console.Write(bits[k]? 1 : 0);
                        if (k == 15)
                        {
                            System.Console.Write(" i = "+i+"\n");
                        }
                    }
                    System.Console.Write(" j = "+j+"\n");
                }
                System.Console.WriteLine();
                System.Console.WriteLine();
                for (int j = 0; j < valueN.GetLength(0); j++)
                {
                    //System.Console.Write(valueN[j] + " ");
                    BitArray b = new BitArray(new int[] { valueN[j] });
                    bool[] bits = new bool[b.Count];
                    b.CopyTo(bits, 0);

                    if (j % (valueN.Count() / 6) == 0)
                    {
         //               System.Console.Write(" j = " + j + "\n");
                    }

                    for (int k = 0; k < bits.GetLength(0); k++)
                    {
                        System.Console.Write(bits[k]? 1 : 0);
                        if (k == 15)
                        {
                            System.Console.Write(" i = "+i+"\n");
                        }
                    }
                    System.Console.Write(" j = "+j+"\n");
                }
                System.Console.WriteLine();
                System.Console.WriteLine();
                System.Console.WriteLine();
                System.Console.WriteLine();
                
            }
        }

        private void SaveResponse(Dictionary<Minutia, Tuple<int[, ,], int[, ,]>> response, List<Minutia> minutiae)
        {
            int[,] oneImage = new int[7 * (response[minutiae[0]].Item1.GetLength(0)) + 7, response[minutiae[0]].Item1.GetLength(1)];
            for (int i = 0; i < response.Count; i++)
            {
                for (int j = 0; j < response[minutiae[i]].Item1.GetLength(0); j++)
                {
                    for (int k = 0; k < response[minutiae[i]].Item1.GetLength(1); k++)
                    {
                        oneImage[k, j] = response[minutiae[i]].Item2[j, k, 0];
                        oneImage[k + 17, j] = response[minutiae[i]].Item1[j, k, 0];
                        oneImage[k + 17 * 2, j] = response[minutiae[i]].Item1[j, k, 1];
                        oneImage[k + 17 * 3, j] = response[minutiae[i]].Item1[j, k, 2];
                        oneImage[k + 17 * 4, j] = response[minutiae[i]].Item1[j, k, 3];
                        oneImage[k + 17 * 5, j] = response[minutiae[i]].Item1[j, k, 4];
                        oneImage[k + 17 * 6, j] = response[minutiae[i]].Item1[j, k, 5];
                    }
                    oneImage[16, j] 
                        = oneImage[17 + 16, j] 
                        = oneImage[16 + 17 * 2, j] 
                        = oneImage[16 + 17 * 3, j] 
                        = oneImage[16 + 17 * 4, j] 
                        = oneImage[16 + 17 * 5, j] 
                        = oneImage[16 + 17 * 6, j] = 253;
                }
                oneImage = Img3DHelper.Normalize(oneImage);
                string path = Path.GetTempPath() + "valueN" + i + ".tif";
                ImageHelper.SaveIntArray(oneImage,path);

                //var bmp = new Bitmap(path);
                //Graphics graphic = Graphics.FromImage(bmp);
                //graphic.DrawLine(Pens.Red,16,0,16,15);
                //graphic.DrawLine(Pens.Red, 16+17, 0,  16+17, 15);
                //graphic.DrawLine(Pens.Red, 16 + 17 * 2, 0, 16 + 17 * 2, 15);
                //graphic.DrawLine(Pens.Red, 16 + 17 * 3, 0, 16 + 17 * 3, 15);
                //graphic.DrawLine(Pens.Red, 16 + 17 * 4, 0, 16 + 17 * 4, 15);
                //graphic.DrawLine(Pens.Red, 16 + 17 * 5, 0, 16 + 17 * 5, 15);
                //graphic.DrawLine(Pens.Red, 16 + 17 * 6, 0, 16 + 17 * 6, 15);
                //graphic.Save();
                //bmp.Save(path, ImageFormat.Tiff);   
                
            }
            //  Graphics graphics = Graphics.FromImage(Resources._104_6);
        }

        [TestMethod]
        public void ProcessImg()
        {
            ImageHelper.SaveArray(ImageEnhancementHelper.EnhanceImage(ImageHelper.LoadImage("C:\\temp\\acd.png")), "C:\\temp\\acd_enh.bmp");
        }
    }
}
