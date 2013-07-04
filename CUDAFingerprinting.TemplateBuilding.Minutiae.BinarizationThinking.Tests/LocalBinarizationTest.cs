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
    }
}
