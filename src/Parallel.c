#include "Project.c"
#include <mpi.h>

Matrix parallelMatrixMultiplication(Matrix m1, Matrix m2)
{
    if (m1.cols != m2.rows)
    {
        printf("m1: %d, %d ", m1.rows, m1.cols);
        printf("m2: %d, %d ", m2.rows, m2.cols);
        perror("Incorrect matrix sizes! In multiplication");
        exit(1);
    }

    Matrix mRes = createMatrix(m1.rows, m2.cols);

    for (size_t i = 0; i < m1.rows; i++)
    {
        for (size_t j = 0; j < m2.cols; j++)
        {
            for (size_t k = 0; k < m1.cols; k++)
            {
                mRes.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }
    freeMatrix(m1);
    freeMatrix(m2);

    return mRes;
}

double parallelTwoLayerPerceptron(double learningRate, int numOfHidden, int epochs, Matrix X, Matrix targets, int argc, char *argv[])
{
    int rank, size, rc;

    MPI_Status status;
    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    Matrix W1 = generateRandomMatrix(numOfHidden, X.rows);
    Matrix W2 = generateRandomMatrix(1, numOfHidden+1);
    for (int i = 0; i < epochs; i++)
    {   
        Matrix hidden = addRowWithOnes(applyFunction(parallelMatrixMultiplication(copy(W1), copy(X)), phi));
        
        Matrix out = applyFunction(parallelMatrixMultiplication(copy(W2), copy(hidden)), phi);
        
        Matrix delta_o = elementwiseMultiplication(
            matrixSubtraction(copy(out), copy(targets)), 
            applyFunction(copy(out), phiDeriv));
        freeMatrix(out);
        Matrix delta_h = elementwiseMultiplication(
            parallelMatrixMultiplication(transpose(copy(W2)), copy(delta_o)), 
            applyFunction(copy(hidden), phiDeriv));
        
        delta_h = removeRow(delta_h, delta_h.rows - 1);
        Matrix gradient_w1 = parallelMatrixMultiplication(delta_h, transpose(copy(X)));
        Matrix gradient_w2 = parallelMatrixMultiplication(delta_o, transpose(hidden));

        Matrix delta_w1 = scalarMultiplication(negativeMatrix(gradient_w1), learningRate);
        Matrix delta_w2 = scalarMultiplication(negativeMatrix(gradient_w2), learningRate);

        W1 = matrixAddition(W1, delta_w1);
        W2 = matrixAddition(W2, delta_w2);
    }
    Matrix hidden = addRowWithOnes(applyFunction(parallelMatrixMultiplication(W1, X), phi));
    Matrix out = applyFunction(parallelMatrixMultiplication(W2, hidden), phi);
    printMatrix(out);

    double mse = 0;
    for (int i = 0; i < targets.cols; i++)
    {
        mse += pow(out.data[0][i] - targets.data[0][i],2);
    }
    mse /= targets.cols; 
    freeMatrix(out);
    if (rank != 0)
    {
        rc = MPI_Send(&mse, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        printf("%f from 0\n", mse);
        for (int i = 1; i < size; i++)
        {
            rc = MPI_Recv(&mse,1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            printf("%f from %d\n", mse, i);
        }
    }

    rc = MPI_Finalize();

    return mse;
}

int main(int argc, char *argv[])
{   
    //Alla körs...
    perror("gaming");
    char *filepath = "data/1dFuncData.txt";
    Matrix input = read1DFuncData(filepath);
    filepath = "data/cringe.txt";
    Matrix input2 = read1DFuncData(filepath);

    double mse = parallelTwoLayerPerceptron(0.001, 20, 10000, input, input2, argc, argv);

    printf("MSE: %f", mse);

    freeMatrix(input2);
    return 0;
}