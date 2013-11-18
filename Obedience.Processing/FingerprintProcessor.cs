using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.Common.Segmentation;
using CUDAFingerprinting.ImageEnhancement.LinearSymmetry;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinning;

namespace Obedience.Processing
{


    public class FingerprintProcessor
    {
        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CUDASegmentator")]
        private static extern void CUDASegmentator(float[] img, int imgWidth, int imgHeight, float weightConstant,
                                                int windowSize, int[] mask, int maskWidth, int maskHight);

        [DllImport("CUDAOrientationField.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "makeOrientationField")]
        private static extern void CUDAMakeOrientationField (float[] img, int imgWidth, int imgHeight, float[] orField, int regionSize, int overlap);

        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "PostProcessing")]
        private static extern void PostProcessing(int[] mask, int maskX, int maskY, int threshold);

        [DllImport("CUDAThining.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CUDAThining")]
        private static extern void CUDAThining(float[] picture, int width, int height, float[] result);

        [DllImport("CUDAMinutiaeMatcher.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void FillDirections();

        [DllImport("CUDAMinutiaeMatcher.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Enhance(float[] image, int width, int height);

        [DllImport("CUDABinarization.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void CudaGlobalBinarization(float border, float[] img_dst, float[] img_src, int width, int height);

        public void LoadTemplates()
        {

        }

        public void ProcessFingerImage(float[] image, int rows, int columns)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            int[,] mask;

            var doubleImage = SegmentImage(image, rows, columns, out mask, true);

            doubleImage = BinarizeImage(doubleImage, rows, columns, true);

            doubleImage = ThinImage(doubleImage, rows, columns, true);
            sw.Stop();
            // temporary
            var path = "C:\\temp\\Tercom_Thinned\\" + Guid.NewGuid() + ".png";

            ImageHelper.SaveArray(doubleImage.Make2D(rows, columns).Select2D(x => (double)x), path);
            Trace.WriteLine(string.Format("Processing of the {0} took {1} ms", path, sw.ElapsedMilliseconds));
            //List<Minutia> minutiae = FindMinutiae(doubleImage, rows, columns);


        }

        //public List<Minutia> FindMinutiae(float[] image, int rows, int columns, bool useCuda = false)
        //{
        //    if (useCuda)
        //    {
        //        return new List<Minutia>();
        //    }

        //    return
        //        MinutiaeDetection.FindBigMinutiae(
        //            MinutiaeDetection.FindMinutiae(image.Select(x => (double) x).ToArray().Make2D(rows, columns)));
        //}

        public float[] ThinImage(float[] image, int rows, int columns, bool useCUDA = false)
        {
            if (useCUDA)
            {
                var result = new float[rows * columns];
                CUDAThining(image, columns, rows, result);
                return result;
            }
            return
                Thining.ThinPicture(image.Make2D(rows, columns).Select2D(x => (double) x))
                    .Select2D(x => (float) x)
                    .Make1D();
        }

        public float[] MakeOrientationField(float[] image, int rows, int columns, int regionSize, int overlap, bool useCUDA = false)
        {
            if (useCUDA)
            {
                int orFieldWidth = columns / (regionSize - overlap);
                int orFieldHeight = rows / (regionSize - overlap);

                var result = new float[orFieldWidth * orFieldHeight];
                CUDAMakeOrientationField(image, columns, rows, result, regionSize, overlap);
                return result;
            }
            return
                OrientationFieldGenerator.GenerateOrientationField(image.Select(x => (int) x)
                                                                        .ToArray()
                                                                        .Make2D(rows, columns))
                                         .Make1D().Select(x => (float) x).ToArray();
        }

        public float[] BinarizeImage(float[] image, int rows, int columns, bool useCUDA = false)
        {
            if (useCUDA)
            {
                Enhance(image, columns, rows);
                var result = new float[rows*columns];
                CudaGlobalBinarization((float)Constants.BinarizationThreshold, result, image, columns, rows);
                return result;
            }
            var newImage = ImageEnhancementHelper.EnhanceImage(image.Make2D(rows, columns).Select2D(x => (double)x));

            return
                GlobalBinarization.Binarization(newImage, Constants.BinarizationThreshold)
                    .Make1D()
                    .Select(x => (float)x)
                    .ToArray();
        }

        public float[] SegmentImage(float[] image, int rows, int columns, out int[,] mask, bool UseCUDA = false)
        {
            if (UseCUDA)
            {
                int maskRows = (int) Math.Ceiling((double) rows/Constants.SegmentationWindowSize);
                int maskColumns = (int) Math.Ceiling((double) columns/Constants.SegmentationWindowSize);
                int[] maskLinearized = new int[maskRows * maskColumns];

                Stopwatch sw = new Stopwatch();
                sw.Start();
                CUDASegmentator(image, columns, rows,
                    (float)Constants.SegmentationWeight, Constants.SegmentationWindowSize, maskLinearized, maskColumns, maskRows);
                sw.Stop();
                mask = maskLinearized.Make2D(maskRows, maskColumns);

                Segmentator.PostProcessing(mask, Constants.SegmentationThreshold);
                var bigMask = Segmentator.GetBigMask(mask, rows, columns,
                    Constants.SegmentationWindowSize);

                return Segmentator.ColorImage(image, rows, columns,bigMask);
            }
            else
            {
                mask = Segmentator.Segmetator(image.Make2D(rows, columns).Select2D(x => (double) x),
                    Constants.SegmentationWindowSize, Constants.SegmentationWeight,
                    Constants.SegmentationThreshold);
                return Segmentator.ColorImage(image.Make2D(rows, columns).Select2D(x => (double) x),
                    Segmentator.GetBigMask(mask, image.GetLength(0), image.GetLength(1),
                        Constants.SegmentationWindowSize)).Make1D().Select(x=>(float)x).ToArray();
            }
        }

        //public List<Minutia> FilterMinutiae(List<Minutia> result, int[,] segment)
        //{
        //    var size = Constants.SegmentationWindowSize;

        //    var maxX = segment.GetLength(0);
        //    var maxY = segment.GetLength(1);

        //    var output = new List<Minutia>();

        //    foreach (var minutia in result)
        //    {
        //        var y = minutia.X/size;
        //        var x = minutia.Y/size;

        //        try
        //        {
        //            if (segment[x, y] == 1)
        //            {

        //                if (x > 0)
        //                {
        //                    if (segment[x - 1, y] == 0) continue;
        //                    if (y > 0) if (segment[x - 1, y - 1] == 0) continue;
        //                    if (y < maxY) if (segment[x - 1, y + 1] == 0) continue;
        //                }
        //                if (x < maxX)
        //                {
        //                    if (segment[x + 1, y] == 0) continue;
        //                    if (y > 0) if (segment[x + 1, y - 1] == 0) continue;
        //                    if (y < maxY) if (segment[x + 1, y + 1] == 0) continue;
        //                }
        //                if (y > 0)
        //                {
        //                    if (segment[x, y - 1] == 0) continue;
        //                }
        //                if (y < maxY)
        //                {
        //                    if (segment[x, y + 1] == 0) continue;
        //                }

        //                output.Add(minutia);
        //            }
        //        }
        //        catch (Exception)
        //        {
                    
        //            throw;
        //        }
               
        //    }

        //    return output;
        //}
    }
}
