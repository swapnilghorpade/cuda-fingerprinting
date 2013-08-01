using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common.ComplexFilters;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.Common.PoreFilter;
using CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Drawing;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageEnhancement.ContextualGabor;
using OrientationFieldGenerator = CUDAFingerprinting.ImageEnhancement.ContextualGabor.OrientationFieldGenerator;

namespace CUDAFingerprinting.TemplateBuilding.Minutiae.BinarizationThinking.Tests
{
    
    
    /// <summary>
    ///Это класс теста для MinutiaeDirectionTest, в котором должны
    ///находиться все модульные тесты MinutiaeDirectionTest
    ///</summary>
    [TestClass()]
    public class MinutiaeDirectionTest
    {
        [DllImport("CUDASegmentation.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void CUDASegmentator(float[] img, int imgWidth, int imgHeight, float weightConstant, int windowSize, int[] mask, int maskWidth, int maskHight);

        private TestContext testContextInstance;

        /// <summary>
        ///Получает или устанавливает контекст теста, в котором предоставляются
        ///сведения о текущем тестовом запуске и обеспечивается его функциональность.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region Дополнительные атрибуты теста
        // 
        //При написании тестов можно использовать следующие дополнительные атрибуты:
        //
        //ClassInitialize используется для выполнения кода до запуска первого теста в классе
        //[ClassInitialize()]
        //public static void MyClassInitialize(TestContext testContext)
        //{
        //}
        //
        //ClassCleanup используется для выполнения кода после завершения работы всех тестов в классе
        //[ClassCleanup()]
        //public static void MyClassCleanup()
        //{
        //}
        //
        //TestInitialize используется для выполнения кода перед запуском каждого теста
        //[TestInitialize()]
        //public void MyTestInitialize()
        //{
        //}
        //
        //TestCleanup используется для выполнения кода после завершения каждого теста
        //[TestCleanup()]
        //public void MyTestCleanup()
        //{
        //}
        //
        #endregion


        /// <summary>
        ///Тест для FindDirection
        ///</summary>


        [TestMethod()]
        public void FindDirectionTest()
        {
            for (int cur = 1; cur <= 10; cur++)
            {
                var bmp = new Bitmap("D:\\MyData\\Documents\\Results\\Samples\\" + cur + ".png");
                double[,] startImg = ImageHelper.LoadImage(bmp);
                int imgHeight = startImg.GetLength(0);
                int imgWidth = startImg.GetLength(1);
                //-------------------------------
                int[] mask = new int[imgHeight*imgWidth];
                int windowSize = 12;
                float WeightConstant = 0.3F;
                int maskHeight = imgHeight/windowSize;
                int maskWidth = imgWidth/windowSize;
                float[] imgToSegmentator = new float[imgHeight*imgWidth];
                for (int i = 0; i < imgHeight; i++)
                    for (int j = 0; j < imgWidth; j++)
                        imgToSegmentator[i*imgWidth + j] = (float) startImg[i, j];

                CUDASegmentator(imgToSegmentator, imgWidth, imgHeight, WeightConstant, windowSize, mask, maskWidth,
                                maskHeight);

                //----------------------------------

                double sigma = 1.4d;
                double[,] smoothing = LocalBinarizationCanny.Smoothing(startImg, sigma);
                double[,] sobel = LocalBinarizationCanny.Sobel(smoothing);
                double[,] nonMax = LocalBinarizationCanny.NonMaximumSupperession(sobel);
                nonMax = GlobalBinarization.Binarization(nonMax, 60);
                nonMax = LocalBinarizationCanny.Inv(nonMax);
                int sizeWin = 16;
                startImg = LocalBinarizationCanny.LocalBinarization(startImg, nonMax, sizeWin, 1.3d);
                startImg = GlobalBinarization.Binarization(startImg, 150);
                //-------------------------------

                startImg = Thining.ThiningPicture(startImg);

                //-------------------------------
                for (int i = 0; i < imgHeight; i++)
                    for (int j = 0; j < imgWidth; j++)
                        if (mask[i/windowSize*maskWidth + j/windowSize] == 0)
                            startImg[i, j] = 255;
                var path1 = Path.GetTempPath() + "lol.png";
                ImageHelper.SaveArray(startImg, path1);
                //-------------------------------

                int[,] BinaryImage = ImageHelper.ConvertDoubleToInt(startImg);
                double[,] OrientationField =
                    CUDAFingerprinting.Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(
                        BinaryImage);

                //-------------------------------

                List<Minutia> Minutiae = BinarizationThinking.MinutiaeDetection.FindMinutiae(startImg);
                //--------------------------------
                for (int i = 0; i < Minutiae.Count; i++)
                {
                    if (mask[Minutiae[i].Y / windowSize * maskWidth + Minutiae[i].X / windowSize] == 0)
                    {
                        Minutiae.Remove(Minutiae[i]);
                        i--;
                    }
                }

                //--------------------------------

                Minutiae = BinarizationThinking.MinutiaeDetection.FindBigMinutiae(Minutiae);

                //-------------------------------

                /*MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage, 1);

                //-------------------------------

                var path = "D:\\MyData\\Documents\\Results\\1\\" + cur + ".png";
                ImageHelper.MarkMinutiaeWithDirections(path1, Minutiae, path);

                //-------------------------------

                MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage, 4);

                //-------------------------------

                path = "D:\\MyData\\Documents\\Results\\2\\" + cur + ".png";
                ImageHelper.MarkMinutiaeWithDirections(path1, Minutiae, path);

                //-------------------------------

                MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage, 2);

                //-------------------------------

                path = "D:\\MyData\\Documents\\Results\\3\\" + cur + ".png";
                ImageHelper.MarkMinutiaeWithDirections(path1, Minutiae, path);*/

                //-------------------------------

                MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage, 3);

                //-------------------------------

                var path = "D:\\MyData\\Documents\\Results\\4\\" + cur + ".png";
                ImageHelper.MarkMinutiaeWithDirections(path1, Minutiae, path);
            }
        }
    }
}
