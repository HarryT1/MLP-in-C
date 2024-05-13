
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct Matrix
{
    int rows;
    int cols;
    double **data;
} Matrix;

Matrix createMatrix(int rows, int cols)
{
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        mat.data[i] = (double *)calloc(cols, sizeof(double));
    }
    return mat;
}

void freeMatrix(Matrix mat)
{
    for (int i = 0; i < mat.rows; i++)
    {
        free(mat.data[i]);
    }
    free(mat.data);
}

Matrix transpose(Matrix m)
{
    Matrix mat = createMatrix(m.cols, m.rows);
    for (int i = 0; i < m.cols; i++)
    {
        for (int j = 0; j < m.rows; j++)
        {
            mat.data[i][j] = m.data[j][i];
        }
    }
    freeMatrix(m);
    return mat;
}

Matrix scalarMultiplication(Matrix m, double scalar)
{
    Matrix mat = createMatrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            mat.data[i][j] = m.data[i][j] * scalar;
        }
    }
    freeMatrix(m);
    return mat;
}

Matrix matrixAddition(Matrix m1, Matrix m2)
{
    if (m1.rows != m2.rows || m1.cols != m2.cols)
    {
        printf("m1: %d, %d ", m1.rows, m1.cols);
        printf("m2: %d, %d ", m2.rows, m2.cols);
        perror("Incorrect matrix sizes! In addition");
        exit(1);
    }
    Matrix mat = createMatrix(m1.rows, m1.cols);
    for (int i = 0; i < m1.rows; i++)
    {
        for (int j = 0; j < m1.cols; j++)
        {
            mat.data[i][j] = m1.data[i][j] + m2.data[i][j];
        }
    }
    freeMatrix(m1);
    freeMatrix(m2);
    return mat;
}

Matrix matrixSubtraction(Matrix m1, Matrix m2)
{
    if (m1.rows != m2.rows || m1.cols != m2.cols)
    {
        printf("m1: %d, %d ", m1.rows, m1.cols);
        printf("m2: %d, %d ", m2.rows, m2.cols);
        perror("Incorrect matrix sizes! In subtraction");
        exit(1);
    }
    Matrix mat = createMatrix(m1.rows, m1.cols);
    for (int i = 0; i < m1.rows; i++)
    {
        for (int j = 0; j < m1.cols; j++)
        {
            mat.data[i][j] = m1.data[i][j] - m2.data[i][j];
        }
    }
    freeMatrix(m1);
    freeMatrix(m2);
    return mat;
}

Matrix addRowWithOnes(Matrix m)
{
    Matrix mat = createMatrix(m.rows + 1, m.cols);
    for (int i = 0; i < m.rows + 1; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            if (i < mat.rows - 1)
            {
                mat.data[i][j] = m.data[i][j];
            }
            else
            {
                mat.data[i][j] = 1;
            }
        }
    }
    freeMatrix(m);
    return mat;
}

Matrix removeRow(Matrix m, int removeRow)
{
    Matrix mat = createMatrix(m.rows - 1, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            if (i < removeRow)
            {
                mat.data[i][j] = m.data[i][j];
            }
            else if (i > removeRow)
            {
                mat.data[i - 1][j] = m.data[i][j];
            }
        }
    }
    freeMatrix(m);
    return mat;
}

Matrix negativeMatrix(Matrix m1)
{

    Matrix negative = createMatrix(m1.rows, m1.cols);

    for (size_t i = 0; i < m1.rows; i++)
    {
        for (size_t j = 0; j < m1.cols; j++)
        {
            negative.data[i][j] = -m1.data[i][j];
        }
    }
    freeMatrix(m1);

    return negative;
}

Matrix matrixMultiplication(Matrix m1, Matrix m2)
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

Matrix elementwiseMultiplication(Matrix m1, Matrix m2)
{
    if (m1.rows != m2.rows || m1.cols != m2.cols)
    {
        printf("m1: %d, %d ", m1.rows, m1.cols);
        printf("m2: %d, %d ", m2.rows, m2.cols);
        perror("Incorrect matrix sizes! In elementwise multiplication");
        exit(1);
    }
    Matrix mat = createMatrix(m1.rows, m1.cols);
    for (int i = 0; i < m1.rows; i++)
    {
        for (int j = 0; j < m1.cols; j++)
        {
            mat.data[i][j] = m1.data[i][j] * m2.data[i][j];
        }
    }
    freeMatrix(m1);
    freeMatrix(m2);
    return mat;
}

Matrix copy(Matrix m)
{
    Matrix mat = createMatrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            mat.data[i][j] = m.data[i][j];
        }
    }
    return mat;
}

Matrix read1DFuncData(char *filepath)
{
    FILE *filepointer;
    filepointer = fopen(filepath, "r");

    if (filepointer == NULL)
    {
        perror("Failed to open when reading");
        exit(1);
    }

    int rows, cols;

    fscanf(filepointer, "%d;", &rows);
    fscanf(filepointer, "%d;", &cols);

    Matrix mat = createMatrix(rows, cols);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            fscanf(filepointer, "%lf,", &mat.data[i][j]);
        }
    }

    fclose(filepointer);

    return mat;
}

double phi(double x)
{
    return 2 / (1 + exp(-x)) - 1;
}
double phiDeriv(double phiOfX)
{
    return ((1 + phiOfX) * (1 - phiOfX)) / 2;
}

Matrix applyFunction(Matrix m, double (*function)(double))
{
    Matrix result = createMatrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            result.data[i][j] = function(m.data[i][j]);
        }
    }
    freeMatrix(m);
    return result;
}

Matrix generateRandomMatrix(int rows, int cols){
    Matrix mat = createMatrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            //Not normal dist
            mat.data[i][j] = (-1 + 2 * (double)rand()/RAND_MAX);
        }
    }
    return mat;
}

void printMatrix(Matrix m){
    printf("%d\n",m.rows);
    printf("%d\n",m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            printf("%.2f ", m.data[i][j]);
        }
        printf("\n");
    }
    
}

double twoLayerPerceptron(double learningRate, int numOfHidden, int epochs, Matrix X, Matrix targets)
{
    Matrix W1 = generateRandomMatrix(numOfHidden, X.rows);
    Matrix W2 = generateRandomMatrix(1, numOfHidden+1);
    for (int i = 0; i < epochs; i++)
    {   
        Matrix hidden = addRowWithOnes(applyFunction(matrixMultiplication(copy(W1), copy(X)), phi));
        
        Matrix out = applyFunction(matrixMultiplication(copy(W2), copy(hidden)), phi);
        
        Matrix delta_o = elementwiseMultiplication(
            matrixSubtraction(copy(out), copy(targets)), 
            applyFunction(copy(out), phiDeriv));
        freeMatrix(out);
        Matrix delta_h = elementwiseMultiplication(
            matrixMultiplication(transpose(copy(W2)), copy(delta_o)), 
            applyFunction(copy(hidden), phiDeriv));
        
        delta_h = removeRow(delta_h, delta_h.rows - 1);
        Matrix gradient_w1 = matrixMultiplication(delta_h, transpose(copy(X)));
        Matrix gradient_w2 = matrixMultiplication(delta_o, transpose(hidden));

        Matrix delta_w1 = scalarMultiplication(negativeMatrix(gradient_w1), learningRate);
        Matrix delta_w2 = scalarMultiplication(negativeMatrix(gradient_w2), learningRate);

        W1 = matrixAddition(W1, delta_w1);
        W2 = matrixAddition(W2, delta_w2);
    }
    Matrix hidden = addRowWithOnes(applyFunction(matrixMultiplication(W1, X), phi));
    Matrix out = applyFunction(matrixMultiplication(W2, hidden), phi);
    printMatrix(out);

    double mse = 0;
    for (int i = 0; i < targets.cols; i++)
    {
        mse += pow(out.data[0][i] - targets.data[0][i],2);
    }
    mse /= targets.cols; 
    freeMatrix(out);
    return mse;
}

// int main(int argc, char const *argv[])
// {   perror("test");
//     char *filepath = "../data/1dFuncData.txt";
//     Matrix input = read1DFuncData(filepath);
//     filepath = "../data/cringe.txt";
//     Matrix input2 = read1DFuncData(filepath);

//     double mse = twoLayerPerceptron(0.001, 20, 10000, input, input2);

//     printf("MSE: %f", mse);

//     return 0;
// }
