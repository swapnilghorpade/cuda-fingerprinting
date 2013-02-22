using System.Numerics;

namespace ComplexFilterQA
{
    static class ConvolutionHelper
    {
        public static double[,] Convolve(double[,] data, double[,] kernel)
        {
            int X = data.GetLength(0);
            int Y = data.GetLength(1);

            int I = kernel.GetLength(0);
            int J = kernel.GetLength(1);

            var result = new double[X,Y];

            var centerI = I/2;
            var centerJ = J/2;

            for(int x=0;x<X;x++)
            {
                for (int y = 0; y < Y; y++)
                {
                    for(int i=-centerI;i<=centerI;i++)
                    {
                        for (int j = -centerJ; j <= centerJ; j++)
                        {
                            var indexX = x + i;
                            if(indexX<0) indexX = 0;
                            if(indexX>=X) indexX = X - 1;
                            var indexY = y + j;
                            if(indexY<0) indexY = 0;
                            if(indexY>=Y) indexY = Y - 1;
                            result[x, y] += kernel[centerI - i, centerJ - j]*data[indexX, indexY];
                        }
                    }
                }
            }
            return result;
        }

        public static Complex[,] ComplexConvolve(Complex[,] data, Complex[,] kernel )
        {
            var dataReal = KernelHelper.GetRealPart(data);
            var dataImaginary = KernelHelper.GetImaginaryPart(data);


            var kernelReal = KernelHelper.GetRealPart(kernel);
            var kernelImaginary = KernelHelper.GetImaginaryPart(kernel);

            var resultRealPart1 = Convolve(dataReal, kernelReal);
            var resultRealPart2 = Convolve(dataImaginary, kernelImaginary);

            var resultImaginaryPart1 = Convolve(dataReal, kernelImaginary);
            var resultImaginaryPart2 = Convolve(dataImaginary, kernelReal);

            return KernelHelper.MakeComplexFromDouble(
                KernelHelper.Subtract(resultRealPart1, resultRealPart2),
                KernelHelper.Add(resultImaginaryPart1, resultImaginaryPart2));
        }
    }
}
