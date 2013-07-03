using System;
using System.Collections.Generic;
using System.Drawing;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUDAFingerprinting.Common.ConvexHull.Test
{
    
    
    /// <summary>
    ///Это класс теста для ConvexHullTest, в котором должны
    ///находиться все модульные тесты ConvexHullTest
    ///</summary>
    [TestClass()]
    public class ConvexHullTest
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
        ///Тест для GetConvexHull
        ///</summary>
        [TestMethod()]
        public void GetConvexHullTest()
        {

            List<Point> arr = SerializationHelper.SerializationHelper.DeserializeObject<List<Point>>(Resources.TestArray2); // TODO: инициализация подходящего значения
            List<Point> expected = SerializationHelper.SerializationHelper.DeserializeObject<List<Point>>(Resources.TestArray1Result); // TODO: инициализация подходящего значения
            List<Point> actual = ConvexHull.GetConvexHull(arr);
            Assert.AreEqual(expected, actual);
           
            Assert.Inconclusive("Проверьте правильность этого метода теста.");
            
        }
    }
}
