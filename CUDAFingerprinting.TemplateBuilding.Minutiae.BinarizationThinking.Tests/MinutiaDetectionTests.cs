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
            var img = ImageHelper.LoadImage(TestResource.Test);
            var path = Path.GetTempPath() + "detection.png";
            ImageHelper.MarkMinutiae(TestResource.Test, MinutiaeDetection.FindMinutiae(img), path);
            //Trace.WriteLine(MinutiaeDetection.FindMinutiae(img));
            //ImageHelper.SaveArray(Detection, path);
            Process.Start(path);
        }
    }
}
