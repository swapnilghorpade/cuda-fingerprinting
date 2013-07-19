using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.PoreFilter;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests
{
    [TestClass]
    public class MinutiaDetectionTests
    {
        [TestMethod]
        public void TestMethod1()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource._104_61globalBinarization150Thinned);
            var path = Path.GetTempPath() + "detection.png";
            ImageHelper.MarkMinutiae(TestResource._104_61globalBinarization150Thinned, MinutiaeDetection.FindMinutiae(img), path);
            //Trace.WriteLine(MinutiaeDetection.FindMinutiae(img));
            //ImageHelper.SaveArray(Detection, path);
            Process.Start(path);
        }
        [TestMethod]
        public void TestMethod2()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double board = 5;
            var thining = Thining.ThiningPicture(GlobalBinarization.Binarization(img, board));
            var path = Path.GetTempPath() + "MinutiaeMarkedThinnedBinarizated"+ board +".png";

            ImageHelper.MarkMinutiae(ImageHelper.SaveArrayToBitmap(thining), MinutiaeDetection.FindMinutiae(thining), path);
            //Trace.WriteLine(MinutiaeDetection.FindMinutiae(img));
            //ImageHelper.SaveArray(Detection, path);
            //ImageHelper.SaveArray(thining, path);
            Process.Start(path);
        }
        [TestMethod]
        public void TestMethod3()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource._104_61globalBinarization150Thinned);
            var path = Path.GetTempPath() + "detection.png";
            ImageHelper.MarkMinutiae(TestResource._104_61globalBinarization150Thinned, MinutiaeDetection.FindBigMinutiae(MinutiaeDetection.FindMinutiae(img)), path);
            //Trace.WriteLine(MinutiaeDetection.FindMinutiae(img));
            //ImageHelper.SaveArray(Detection, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod4()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource.MinutiaeBigTest);
            var path = Path.GetTempPath() + "detection.png";
            ImageHelper.MarkMinutiae(TestResource.MinutiaeBigTest, MinutiaeDetection.FindBigMinutiae(MinutiaeDetection.FindMinutiae(img)), path);
            //Trace.WriteLine(MinutiaeDetection.FindMinutiae(img));
            //ImageHelper.SaveArray(Detection, path);
            Process.Start(path);
        }
        [TestMethod]
        public void TestMethod5()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            //ImageHelper.SaveImageAsBinaryFloat("C:\\cuda-fingerprinting\\CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests\\Resources\\104_61globalBinarization150.png", "C:\\temp\\104_6_Binarizated.bin");
            ImageHelper.SaveBinaryAsImage("C:\\temp\\104_6_BinarizatedThinnedMinutiaeMatchedCUDA.bin", "C:\\cuda-fingerprinting\\104_6_BinarizatedThinnedMinutiaeMatchedCUDA.png", true);
            //ImageHelper.SaveIntArray();
 
            Process.Start("C:\\cuda-fingerprinting\\104_6_BinarizatedThinnedMinutiaeMatchedCUDA.png");
        }
        [TestMethod]
        public void TestMethod6()
        {
            double[,] img = ImageHelper.LoadImage("C:\\Users\\Mirza\\Documents\\enh_img\\81_7.png");
            double sigma = 1.4d;
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            int sizeWin = 16;
            double[,] resImg = LocalBinarizationCanny.LocalBinarization(img, nonMax, sizeWin, 1.3d);
            PoreFilter.DeletePores(resImg);
            PoreFilter.DeletePores(resImg);
            var path = Path.GetTempPath() + "BinarizatedPoreFiltred81_7.png";
            ImageHelper.SaveArray(resImg, path);
            Process.Start(path);
            var path2 = Path.GetTempPath() + "Thinned81_7.png";
            var resImg2 = Thining.ThiningPicture(resImg);
            ImageHelper.SaveArray(resImg2, path2);
            Process.Start(path2);
            var list = MinutiaeDetection.FindMinutiae(resImg2);
            var list2 = MinutiaeDetection.FindBigMinutiae(list);
            var path3 = Path.GetTempPath() + "MinutiaeMatchedTest81_7.png";
            ImageHelper.MarkMinutiae(path2, list2, path3);
        }

        [TestMethod]
        public void TestMethod7()
        {
            double[,] img = ImageHelper.LoadImage("C:\\Users\\Mirza\\Documents\\enh_img\\90_3.png");
            double sigma = 1.4d;
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            int sizeWin = 16;
            double[,] resImg = LocalBinarizationCanny.LocalBinarization(img, nonMax, sizeWin, 1.3d);
            PoreFilter.DeletePores(resImg);
            PoreFilter.DeletePores(resImg);
            var path = Path.GetTempPath() + "BinarizatedPoreFiltred90_3.png";
            ImageHelper.SaveArray(resImg, path);
            Process.Start(path);
            var path2 = Path.GetTempPath() + "Thinned90_3.png";
            var resImg2 = Thining.ThiningPicture(resImg);
            ImageHelper.SaveArray(resImg2, path2);
            Process.Start(path2);
            var list = MinutiaeDetection.FindMinutiae(resImg2);
            var list2 = MinutiaeDetection.FindBigMinutiae(list);
            var path3 = Path.GetTempPath() + "MinutiaeMatchedTest90_3.png";
            ImageHelper.MarkMinutiae(path2, list2, path3);

            Process.Start(path3);
        }
        [TestMethod]
        public void TestMethod8()
        {
            double[,] img = ImageHelper.LoadImage("C:\\Users\\Mirza\\Documents\\enh_img\\85_1.png");
            double sigma = 1.4d;
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            int sizeWin = 16;
            double[,] resImg = LocalBinarizationCanny.LocalBinarization(img, nonMax, sizeWin, 1.3d);
            PoreFilter.DeletePores(resImg);
            PoreFilter.DeletePores(resImg);
            var path = Path.GetTempPath() + "BinarizatedPoreFiltred85_1.png";
            ImageHelper.SaveArray(resImg, path);
            Process.Start(path);
            var path2 = Path.GetTempPath() + "Thinned85_1.png";
            var resImg2 = Thining.ThiningPicture(resImg);
            ImageHelper.SaveArray(resImg2, path2);
            Process.Start(path2);
            var list = MinutiaeDetection.FindMinutiae(resImg2);
            var list2 = MinutiaeDetection.FindBigMinutiae(list);
            var path3 = Path.GetTempPath() + "MinutiaeMatchedTest85_1.png";
            ImageHelper.MarkMinutiae(path2, list2, path3);

            Process.Start(path3);
        }
        [TestMethod]
        public void TestMethod9()
        {
            double[,] img = ImageHelper.LoadImage("C:\\Users\\Mirza\\Documents\\enh_img\\81_4.png");
            double sigma = 1.4d;
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            int sizeWin = 16;
            double[,] resImg = LocalBinarizationCanny.LocalBinarization(img, nonMax, sizeWin, 1.3d);
            PoreFilter.DeletePores(resImg);
            PoreFilter.DeletePores(resImg);
            var path = Path.GetTempPath() + "BinarizatedPoreFiltred81_4.png";
            ImageHelper.SaveArray(resImg, path);
            Process.Start(path);
            var path2 = Path.GetTempPath() + "Thinned81_4.png";
            var resImg2 = Thining.ThiningPicture(resImg);
            ImageHelper.SaveArray(resImg2, path2);
            Process.Start(path2);
            var list = MinutiaeDetection.FindMinutiae(resImg2);
            var list2 = MinutiaeDetection.FindBigMinutiae(list);
            var path3 = Path.GetTempPath() + "MinutiaeMatchedTest81_4.png";
            ImageHelper.MarkMinutiae(path2, list2, path3);

            Process.Start(path3);
        }
        [TestMethod]
        public void TestMethod10()
        {
            double[,] img = ImageHelper.LoadImage("C:\\Users\\Mirza\\Documents\\enh_img\\81_8.png");
            double sigma = 1.4d;
            double[,] smoothing = LocalBinarizationCanny.Smoothing(img, sigma);
            double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
            double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
            nonMax = GlobalBinarization.Binarization(nonMax, 60);
            nonMax = LocalBinarizationCanny.Inv(nonMax);
            int sizeWin = 16;
            double[,] resImg = LocalBinarizationCanny.LocalBinarization(img, nonMax, sizeWin, 1.3d);
            PoreFilter.DeletePores(resImg);
            PoreFilter.DeletePores(resImg);
            var path = Path.GetTempPath() + "BinarizatedPoreFiltred81_8.png";
            ImageHelper.SaveArray(resImg, path);
            Process.Start(path);
            var path2 = Path.GetTempPath() + "Thinned81_8.png";
            var resImg2 = Thining.ThiningPicture(resImg);
            ImageHelper.SaveArray(resImg2, path2);
            Process.Start(path2);
            var list = MinutiaeDetection.FindMinutiae(resImg2);
            var list2 = MinutiaeDetection.FindBigMinutiae(list);
            var path3 = Path.GetTempPath() + "MinutiaeMatchedTest81_8.png";
            ImageHelper.MarkMinutiae(path2, list2, path3);

            Process.Start(path3);
        }

    }
}
