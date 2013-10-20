using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.IO;
using System.Diagnostics;
//using CUDAFingerprinting.ImageEnhancement.ContextualGabor;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinning.Tests
{
    [TestClass]
    public class GlobalBinarizationTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            double board = 50d;
            var img = ImageHelper.LoadImage(TestResource._104_6_ench);
            double[,] binarization = GlobalBinarization.Binarization(img, board);
            var path = Path.GetTempPath() + "globalBinarization" + board + ".png";
            ImageHelper.SaveArray(binarization, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod2()
        {
            double board = 200d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] binarization = GlobalBinarization.Binarization(img, board);
            var path = Path.GetTempPath() + "globalBinarization" + board + ".png";
            ImageHelper.SaveArray(binarization, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod3()
        {
            double board = 150d;
            var img = ImageHelper.LoadImage(TestResource._104_6);
            double[,] binarization = GlobalBinarization.Binarization(img, board);
            var path = Path.GetTempPath() + "globalBinarization" + board + ".png";
            ImageHelper.SaveArray(binarization, path);
            Process.Start(path);
        }

        [TestMethod]
        public void TestMethod4()
        {
            double board = 153d;
            var img = ImageHelper.LoadImage(TestResource._110_6);
            double[,] binarization = GlobalBinarization.Binarization(img, board);
            var path = Path.GetTempPath() + "globalBinarization" + board + ".png";
            ImageHelper.SaveArray(binarization, path);
            Process.Start(path);
        }

        //[TestMethod]
        //public void TestMethodCudaToBin()
        //{
        //    ImageHelper.SaveImageAsBinaryFloat("C:\\temp\\104_6_ench.png", "C:\\temp\\104_6.bin");
        //}

        //[TestMethod]
        //public void TestMethodCudaToImg()
        //{
        //    ImageHelper.SaveBinaryAsImage("C:\\temp\\104_6_2.bin", "C:\\temp\\104_6_localBinar60.png", true);
        //}
    }
}