using System;
using System.Diagnostics;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Obedience.Processing;

namespace Obedience.Tests
{
    [TestClass]
    public class ProcessingTests
    {
        [TestMethod]
        public void TestSegmentation()
        {
            for (int i = 0; i < 20; i++)
            {
                var fp = new FingerprintProcessor();

                int[,] mask;

                var image = ImageHelper.LoadImage(Resources.SampleFinger1);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                var result = fp.SegmentImage(image, out mask);

                sw.Stop();
                Trace.WriteLine("Segmentation with CPU took " + sw.ElapsedMilliseconds + " ms");

                var path = Constants.Path + Guid.NewGuid() + ".png";

                ImageHelper.SaveArray(result, path);

                // should result in the cropped fp
                Process.Start(path);
            }
        }

        [TestMethod]
        public void TestCUDASegmentation()
        {
            for (int i = 0; i < 20; i++)
            {
                var fp = new FingerprintProcessor();

                int[,] mask;

                var image = ImageHelper.LoadImage(Resources.SampleFinger1);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                var result = fp.SegmentImage(image, out mask, true);

                sw.Stop();
                Trace.WriteLine("Segmentation with GPU took " + sw.ElapsedMilliseconds + " ms");

                var path = Constants.Path + Guid.NewGuid() + ".png";

                ImageHelper.SaveArray(result, path);

                // should result in the cropped fp
                Process.Start(path);
            }
        }

        //[TestMethod]
        //public void TestSegmentationPlusBigun()
        //{
        //    var fp = new FingerprintProcessor();

        //    int[,] mask;

        //    var result = fp.SegmentImage(ImageHelper.LoadImage(Resources.SampleFinger1), out mask);

        //    var path = Constants.Path + Guid.NewGuid() + ".png";

        //    ImageHelper.SaveArray(result, path);

        //    Process.Start(path);
        //}
    }
}
