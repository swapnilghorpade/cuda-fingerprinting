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

                int rows = image.GetLength(0);
                int columns = image.GetLength(1);

                var src = image.Select2D(x => (float) x).Make1D();

                Stopwatch sw = new Stopwatch();
                sw.Start();

                var result = fp.SegmentImage(src, rows, columns, out mask, true);

                sw.Stop();
                Trace.WriteLine("Segmentation with GPU took " + sw.ElapsedMilliseconds + " ms");

                var path = Constants.Path + Guid.NewGuid() + ".png";

                ImageHelper.SaveArray(result.Make2D(rows,columns).Select2D(x=>(double)x), path);

                // should result in the cropped fp
                Process.Start(path);
            }
        }

        [TestMethod]
        public void TestSegmentationPlusBigunPlusGlobalBinarization()
        {
            for (int i = 0; i < 20; i++)
            {
                var fp = new FingerprintProcessor();

                int[,] mask;

                var image = ImageHelper.LoadImage(Resources.SampleFinger1);

                int rows = image.GetLength(0);
                int columns = image.GetLength(1);

                var src = image.Select2D(x => (float) x).Make1D();

                var result = fp.SegmentImage(src, rows, columns, out mask, true);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                result = fp.BinarizeImage(result, rows, columns, true);


                sw.Stop();
                Trace.WriteLine("Binarization with GPU took " + sw.ElapsedMilliseconds + " ms");

                var path = Constants.Path + Guid.NewGuid() + ".png";

                ImageHelper.SaveArray(result.Make2D(rows, columns).Select2D(x => (double) x), path);

                Process.Start(path);
            }
        }

        //[TestMethod]
        //public void TestSegmentationPlusBigunPlusGlobalBinarizationPlusThinning()
        //{
        //    var fp = new FingerprintProcessor();

        //    int[,] mask;

        //    var result = fp.SegmentImage(ImageHelper.LoadImage(Resources.SampleFinger1), out mask);

        //    result = fp.BinarizeImage(result);

        //    result = fp.ThinImage(result);

        //    var path = Constants.Path + Guid.NewGuid() + ".png";

        //    ImageHelper.SaveArray(result, path);

        //    Process.Start(path);
        //}

        //[TestMethod]
        //public void TestFullCycleUpToMinutiae()
        //{
        //    var fp = new FingerprintProcessor();

        //    int[,] mask;

        //    var segment = fp.SegmentImage(ImageHelper.LoadImage(Resources.SampleFinger1), out mask);
        //    Stopwatch sw = new Stopwatch();
            
        //    var result = fp.BinarizeImage(segment);
        //    sw.Start();
        //    result = fp.ThinImage(result, true);

        //    //var minutiae = fp.FindMinutiae(result);

        //    //minutiae = fp.FilterMinutiae(minutiae, mask);
        //    sw.Stop();
        //    var path = Constants.Path + Guid.NewGuid() + ".png";
        //    ImageHelper.SaveArray(result, path);
        //    //ImageHelper.MarkMinutiae(Resources.SampleFinger1, minutiae, path);

        //    Process.Start(path);
        //}

        [TestMethod]
        public void TestArrayTransformations()
        {
            var path = Constants.Path + Guid.NewGuid() + ".png";
            ImageHelper.SaveArrayAndOpen(ImageHelper.LoadImage(Resources.SampleFinger1).Make1D().Make2D(Resources.SampleFinger1.Height, Resources.SampleFinger1.Width), path);
        }
    }
}
