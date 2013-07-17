using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;
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
    }
}
