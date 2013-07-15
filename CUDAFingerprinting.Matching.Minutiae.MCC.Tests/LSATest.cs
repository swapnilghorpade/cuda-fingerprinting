using CUDAFingerprinting.Matching.Minutiae.MCC;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using CUDAFingerprinting.Common.SerializationHelper;
namespace CUDAFingerprinting.Matching.Minutiae.MCC.Tests
{
    
    
    /// <summary>
    ///Это класс теста для LSATest, в котором должны
    ///находиться все модульные тесты LSATest
    ///</summary>
    [TestClass()]
    public class LSATest
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
        ///Тест для GetScore
        ///</summary>
        [TestMethod()]
        public void GetScoreTest()
        {
            double[,] Gamma = BinarySerializationHelper.DeserializeObject<double[,]>(Resources.Sample3);
            int np = 8;
            double expected = Double.Parse(Resources.LSAAnswer3);
            double actual = LSA.GetScore(Gamma, np);
            Assert.AreEqual(expected, actual);
        }
    }
}
