using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.Segmentation;
using CUDAFingerprinting.ImageEnhancement.LinearSymmetry;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;

namespace Obedience.Processing
{
    public class FingerprintProcessor
    {
        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CUDASegmentator")]
        private static extern void CUDASegmentator(float[] img, int imgWidth, int imgHeight, float weightConstant,
                                                int windowSize, int[] mask, int maskWidth, int maskHight);

        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "PostProcessing")]
        private static extern void PostProcessing(int[] mask, int maskX, int maskY, int threshold);

        public void LoadTemplates()
        {

        }

        public void ProcessFingerImage(int[,] image)
        {
            var doubleImage = image.Select2D(x => (double) x);

            int[,] mask;

            doubleImage = SegmentImage(doubleImage, out mask);

            doubleImage = BinarizeImage(doubleImage);

            doubleImage = ThinImage(doubleImage);

            List<Minutia> minutiae = FindMinutiae(doubleImage);


        }

        public List<Minutia> FindMinutiae(double[,] doubleImage, bool useCuda = false)
        {
            if (useCuda)
            {
                return new List<Minutia>();
            }

            return MinutiaeDetection.FindBigMinutiae(MinutiaeDetection.FindMinutiae(doubleImage));
        }

        public double[,] ThinImage(double[,] doubleImage, bool useCUDA = false)
        {
            if (useCUDA)
            {
                return doubleImage; // TODO: Replace
            }
            return Thining.ThinPicture(doubleImage);
        }

        public double[,] BinarizeImage(double[,] image, bool useCUDA = false)
        {
            if (useCUDA)
            {
                // TODO: make correct
                return image;
            }
            var newImage = ImageEnhancementHelper.EnhanceImage(image);

            return GlobalBinarization.Binarization(newImage, Constants.BinarizationThreshold);
        }

        public double[,] SegmentImage(double[,] image, out int[,] mask, bool UseCUDA = false)
        {
            if (UseCUDA)
            {
                int maskX = (int) Math.Ceiling((double) image.GetLength(0)/Constants.SegmentationWindowSize);
                int maskY = (int) Math.Ceiling((double) image.GetLength(1)/Constants.SegmentationWindowSize);
                int[] maskLinearized = new int[maskX*maskY];
                float[] oneDimensionalBinaryImg = new float[image.GetLength(0)*image.GetLength(1)];

                for (int i = 0; i < image.GetLength(0); i++)
                {
                    for (int j = 0; j < image.GetLength(1); j++)
                    {
                        oneDimensionalBinaryImg[j*image.GetLength(0) + i] = (float) image[i, j];
                    }
                }

                CUDASegmentator(oneDimensionalBinaryImg, image.GetLength(0), image.GetLength(1),
                    (float) Constants.SegmentationWeight, Constants.SegmentationWindowSize, maskLinearized, maskX, maskY);

                var imageX = image.GetLength(0);
                var imageY = image.GetLength(1);

                mask = new int[maskX, maskY];
                for (int x = 0; x < maskX; x++)
                {
                    for (int y = 0; y < maskY; y++)
                    {
                        mask[x, y] = maskLinearized[y*maskX + x];
                    }
                }

                Segmentator.PostProcessing(mask, Constants.SegmentationThreshold);
            }
            else
                mask = Segmentator.Segmetator(image, Constants.SegmentationWindowSize, Constants.SegmentationWeight,
                    Constants.SegmentationThreshold);

            return Segmentator.ColorImage(image,
                Segmentator.GetBigMask(mask, image.GetLength(0), image.GetLength(1), Constants.SegmentationWindowSize));
        }

        public List<Minutia> FilterMinutiae(List<Minutia> result, int[,] segment)
        {
            var size = Constants.SegmentationWindowSize;

            var maxX = segment.GetLength(0);
            var maxY = segment.GetLength(1);

            var output = new List<Minutia>();

            foreach (var minutia in result)
            {
                var y = minutia.X/size;
                var x = minutia.Y/size;

                try
                {
                    if (segment[x, y] == 1)
                    {

                        if (x > 0)
                        {
                            if (segment[x - 1, y] == 0) continue;
                            if (y > 0) if (segment[x - 1, y - 1] == 0) continue;
                            if (y < maxY) if (segment[x - 1, y + 1] == 0) continue;
                        }
                        if (x < maxX)
                        {
                            if (segment[x + 1, y] == 0) continue;
                            if (y > 0) if (segment[x + 1, y - 1] == 0) continue;
                            if (y < maxY) if (segment[x + 1, y + 1] == 0) continue;
                        }
                        if (y > 0)
                        {
                            if (segment[x, y - 1] == 0) continue;
                        }
                        if (y < maxY)
                        {
                            if (segment[x, y + 1] == 0) continue;
                        }

                        output.Add(minutia);
                    }
                }
                catch (Exception)
                {
                    
                    throw;
                }
               
            }

            return output;
        }
    }
}
