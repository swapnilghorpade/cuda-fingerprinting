using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests
{
    [TestClass]
    public class ThiningTests
    {
        [TestMethod]
        public void TestMethod1()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource.ThiningImageTest);
            var path = Path.GetTempPath() + "thininig.png";
            var thining = Thining.ThiningPicture(img);
            ImageHelper.SaveArray(thining, path);
            Process.Start(path);
        }
        [TestMethod]
        public void TestMethod2()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource.ThiningImageTest2);
            var path = Path.GetTempPath() + "thininig.png";
            var thining = Thining.ThiningPicture(img);
            ImageHelper.SaveArray(thining, path);
            Process.Start(path);
        }
        [TestMethod]
        public void TestMethod3()
        {
            //var img = ImageHelper.LoadImage(TestResource._104_6);
            var img = ImageHelper.LoadImage(TestResource.ThiningImageTest3);
            var path = Path.GetTempPath() + "thininig.png";
            var thining = Thining.ThiningPicture(img); 
            ImageHelper.SaveArray(thining, path);
            Process.Start(path);
        }
    }
}
