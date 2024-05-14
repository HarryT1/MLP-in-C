#include "ParallelMatrix.c"
#include <mpi.h>

#define MULTIPLICATIONSREQPEREPOCH 2

double parallelTwoLayerPerceptron(double learningRate, int numOfHidden, int epochs, Matrix X, Matrix targets, int rank, int size)
{
    double mse;
    if (rank == 0)
    {   
        X = addRowWithOnes(X);
        Matrix W1 = generateRandomMatrix(numOfHidden, X.rows);
        Matrix W2 = generateRandomMatrix(1, numOfHidden + 1);
        for (int i = 0; i < epochs; i++)
        {
            Matrix hidden = addRowWithOnes(applyFunction(parallelMatrixMult(copy(W1), copy(X), rank, size), phi));

            Matrix out = applyFunction(matrixMultiplication(copy(W2), copy(hidden)), phi);

            Matrix delta_o = elementwiseMultiplication(
                matrixSubtraction(copy(out), copy(targets)),
                applyFunction(copy(out), phiDeriv));
            freeMatrix(out);

            
            Matrix delta_h = elementwiseMultiplication(
                parallelMatrixMult(transpose(copy(W2)), copy(delta_o), rank, size),
                applyFunction(copy(hidden), phiDeriv));

            delta_h = removeRow(delta_h, delta_h.rows - 1);
            
            Matrix gradient_w1 = matrixMultiplication(delta_h, transpose(copy(X)));

            Matrix gradient_w2 = matrixMultiplication(delta_o, transpose(hidden));

            Matrix delta_w1 = scalarMultiplication(negativeMatrix(gradient_w1), learningRate);
            Matrix delta_w2 = scalarMultiplication(negativeMatrix(gradient_w2), learningRate);

            W1 = matrixAddition(W1, delta_w1);
            W2 = matrixAddition(W2, delta_w2);
        }
        Matrix hidden = addRowWithOnes(applyFunction(parallelMatrixMult(W1, X, rank, size), phi));
        Matrix out = applyFunction(matrixMultiplication(W2, hidden), phi);
        printMatrix(out);

        mse = 0;
        for (int i = 0; i < targets.cols; i++)
        {
            mse += pow(out.data[0][i] - targets.data[0][i], 2);
        }
        mse /= targets.cols;
        freeMatrix(out);
    }
    else
    {
        Matrix A, B;
        for (int i = 0; i < epochs; i++)
        {
            for (int j = 0; j < MULTIPLICATIONSREQPEREPOCH; j++)
            {
                parallelMatrixMult(A, B, rank, size);  
            }
        }
        // one more
        parallelMatrixMult(A, B, rank, size);
    }

    return mse;
}

int main(int argc, char *argv[])
{
    double learningRate = 0.001;
    int numOfHidden = 25;
    int epochs = 10;

    int rank, size, rc;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        char *filepath = "data/input(1).txt";
        Matrix input = read1DFuncData(filepath);
        filepath = "data/output(1).txt";
        Matrix input2 = read1DFuncData(filepath);
        clock_t start = clock();
        double mse = parallelTwoLayerPerceptron(learningRate, numOfHidden, epochs, input, input2, rank, size);
        clock_t difference = clock() - start;
        printf("Time: %f\n", difference/(float)CLOCKS_PER_SEC);
        freeMatrix(input2);
        printf("MSE: %f\n", mse);
    }
    else
    {
        Matrix A, B;
        parallelTwoLayerPerceptron(learningRate, numOfHidden, epochs, A, B, rank, size);
    }
    MPI_Finalize();
    
    return 0;
}