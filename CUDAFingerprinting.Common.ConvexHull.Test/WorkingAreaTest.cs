﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;
using System.Collections.Generic;

namespace CUDAFingerprinting.Common.ConvexHull.Test
{
    
    
    /// <summary>
    ///Это класс теста для WorkingAreaTest, в котором должны
    ///находиться все модульные тесты WorkingAreaTest
    ///</summary>
    [TestClass()]
    public class WorkingAreaTest
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
        ///Тест для BuildWorkingArea
        ///</summary>
        [TestMethod()]
        public void BuildWorkingAreaTest()
        {
            /*int radius = 2; // TODO: инициализация подходящего значения
            int rows = 10; // TODO: инициализация подходящего значения
            int columns = 10; // TODO: инициализация подходящего значения
            List<Point> Minutiae = SerializationHelper.XmlSerializationHelper.DeserializeObject<List<Point>>(Resources.Sample1);
            bool[,] actual = WorkingArea.BuildWorkingArea(Minutiae, radius, rows, columns);
            bool[,] expected = SerializationHelper.BinarySerializationHelper.DeserializeObject<bool[,]>(Resources.WorkingArea1Answer);
            */
        }
    }
}
