using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.IO;
using System.Diagnostics;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests
{
    [TestClass]
    public class LocalBinarizationTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            double sigma = 1.4d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            var path = Path.GetTempPath() + "localSmo" + sigma + ".png";
            ImageHelper.SaveArray(smoothing, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod2()
        {
            double sigma = 1.41d;
            var img = ImageHelper.LoadImage(TestResource.Valve_original__1_);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            var path = Path.GetTempPath() + "localSol" + sigma + ".png";
            ImageHelper.SaveArray(sobel, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod3()
        {
            double sigma = 1.41d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            var path = Path.GetTempPath() + "localSol" + sigma + ".png";
            ImageHelper.SaveArray(sobel, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod4()
        {
            double sigma = 1.41d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            var path = Path.GetTempPath() + "localSol" + sigma + ".png";
            ImageHelper.SaveArray(nonMax, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod5()
        {
            double sigma = 1.4d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 50);
            var path = Path.GetTempPath() + "localSol" + sigma + ".png";
            ImageHelper.SaveArray(nonMax, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod6()
        {
            double sigma = 1.4d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 50);
            nonMax = LocalBinarizationCanny.Traceroute(nonMax);
            var path = Path.GetTempPath() + "localSol" + sigma + ".png";
            ImageHelper.SaveArray(nonMax, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod7()
        {
            double sigma = 1.4d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 50);
            nonMax = LocalBinarizationCanny.Traceroute(nonMax);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            var path = Path.GetTempPath() + "localSol" + sigma + ".png";
            ImageHelper.SaveArray(nonMax, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestLocalBinarization()
        {
            double sigma = 1.4d;
            var img = ImageHelper.LoadImage(TestResource._104_6_ench);
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);

            int sizeWin = 16;
            double[,] resImg = LocalBinarizationCanny.LocalBinarization(img, nonMax, sizeWin, 1.3d);

            var path = Path.GetTempPath() + "localSol" + sizeWin + ".png";
            ImageHelper.SaveArray(resImg, path);
            Process.Start(path);
        }

        //for CUDA
        [TestMethod]
        public void TestMethodSobel()
        {
            var img = ImageHelper.LoadImage(TestResource._104_6_ench);
            double[,] sobel = LocalBinarizationCanny.Sobel(img);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            double[,] gl = GlobalBinarization.Binarization(nonMax, 60);
            gl = LocalBinarizationCanny.Inv(gl);
            var path = Path.GetTempPath() + "localBinSobel11.png";
            ImageHelper.SaveArray(gl, path);
            Process.Start(path);
        }


        [TestMethod]
        public void TestMethodCudaToBin()
        {
            for (int i = 1; i < 101; i++)
            {
                for (int j = 1; j < 9; j++)
                {
                    ImageHelper.SaveImageAsBinaryFloat("D:\\learn\\learn\\summer_6_5\\from school\\FVC2000\\Dbs\\Db1_a\\" + i + "_" + j + ".tif",
                        "D:\\temp\\binFromImg\\" + i + "_" + j + ".bin");
                }
            }
        }

        [TestMethod]
        public void TestMethodCudaToImg()
        {
            ImageHelper.SaveBinaryAsImage("C:\\temp\\104_6_2.bin", "C:\\temp\\104_6_localBinar60.png", true);
        }

    }
}
