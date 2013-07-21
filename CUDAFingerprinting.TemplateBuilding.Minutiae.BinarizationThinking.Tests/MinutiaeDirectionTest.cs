using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        public void FindDirectionVersion1Test()
        {
            var bmp = new Bitmap(TestResource._104_61globalBinarization150Thinned);
            int[,] BinaryImage = ImageHelper.LoadImageAsInt(bmp);
            double[,] OrientationField = CUDAFingerprinting.Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(BinaryImage);
            for (int i = 0 ; i < OrientationField.GetLength(0); i++)
                for (int j = 0 ; j < OrientationField.GetLength(1); j++)
                    if (OrientationField[i, j] < 0)
                        OrientationField[i, j] += Math.PI;
            double[,] temp = ImageHelper.LoadImage(bmp);
            var path = Path.GetTempPath() + "direction1.png";
            List<Minutia> Minutiae = BinarizationThinking.MinutiaeDetection.FindMinutiae(temp);
            Minutiae = BinarizationThinking.MinutiaeDetection.FindBigMinutiae(Minutiae);
            MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage, 1);
            ImageHelper.MarkMinutiaeWithDirections(TestResource._104_61globalBinarization150Thinned, Minutiae, path);
            Process.Start(path);
        }
        [TestMethod()]
        public void FindDirectionVersion2Test()
        {
            var bmp = new Bitmap(TestResource._104_61globalBinarization150Thinned);
            int[,] BinaryImage = ImageHelper.LoadImageAsInt(bmp);
            double[,] OrientationField = CUDAFingerprinting.Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(BinaryImage);
            for (int i = 0; i < OrientationField.GetLength(0); i++)
                for (int j = 0; j < OrientationField.GetLength(1); j++)
                    if (OrientationField[i, j] < 0)
                        OrientationField[i, j] += Math.PI;
            double[,] temp = ImageHelper.LoadImage(bmp);
            var path = Path.GetTempPath() + "direction2.png";
            List<Minutia> Minutiae = BinarizationThinking.MinutiaeDetection.FindMinutiae(temp);
            Minutiae = BinarizationThinking.MinutiaeDetection.FindBigMinutiae(Minutiae);
            MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage, 2);
            ImageHelper.MarkMinutiaeWithDirections(TestResource._104_61globalBinarization150Thinned, Minutiae, path);
            Process.Start(path);
        }
        [TestMethod()]
        public void FindDirectionVersion3Test()
        {
            var bmp = new Bitmap(TestResource._104_61globalBinarization150Thinned);
            int[,] BinaryImage = ImageHelper.LoadImageAsInt(bmp);
            double[,] OrientationField = CUDAFingerprinting.Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(BinaryImage);
            for (int i = 0; i < OrientationField.GetLength(0); i++)
                for (int j = 0; j < OrientationField.GetLength(1); j++)
                    if (OrientationField[i, j] < 0)
                        OrientationField[i, j] += Math.PI;
            double[,] temp = ImageHelper.LoadImage(bmp);
            var path = Path.GetTempPath() + "direction3.png";
            List<Minutia> Minutiae = BinarizationThinking.MinutiaeDetection.FindMinutiae(temp);
            Minutiae = BinarizationThinking.MinutiaeDetection.FindBigMinutiae(Minutiae);
            MinutiaeDirection.FindDirectionVersion2(OrientationField, 16, Minutiae, BinaryImage);
            ImageHelper.MarkMinutiaeWithDirections(TestResource._104_61globalBinarization150Thinned, Minutiae, path);
            Process.Start(path);
        }
        [TestMethod()]
        public void FindDirectionVersion4Test()
        {
            var bmp = new Bitmap(TestResource._104_61globalBinarization150Thinned);
            int[,] BinaryImage = ImageHelper.LoadImageAsInt(bmp);
            double[,] OrientationField = CUDAFingerprinting.Common.OrientationField.OrientationFieldGenerator.GenerateOrientationField(BinaryImage);
            for (int i = 0; i < OrientationField.GetLength(0); i++)
                for (int j = 0; j < OrientationField.GetLength(1); j++)
                    if (OrientationField[i, j] < 0)
                        OrientationField[i, j] += Math.PI;
            double[,] temp = ImageHelper.LoadImage(bmp);
            var path = Path.GetTempPath() + "direction4.png";
            List<Minutia> Minutiae = BinarizationThinking.MinutiaeDetection.FindMinutiae(temp);
            Minutiae = BinarizationThinking.MinutiaeDetection.FindBigMinutiae(Minutiae);
            MinutiaeDirection.FindDirection(OrientationField, 16, Minutiae, BinaryImage,4);
            ImageHelper.MarkMinutiaeWithDirections(TestResource._104_61globalBinarization150Thinned, Minutiae, path);
            Process.Start(path);
        }
    }
}
