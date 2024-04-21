
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
        mat.data[i] = (double *)malloc(cols * sizeof(double));
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
    return mat;
}

Matrix addRowWithOnes(Matrix m)
{
    Matrix mat = createMatrix(m.rows + 1, m.cols);
    for (int i = 0; i < m.rows + 1; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            if (i < m.rows - 1)
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
double phiDeriv(double x)
{
    return ((1 + phi(x)) * (1 - phi(x))) / 2;
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
            //Not normal, change later 😎
            mat.data[i][j] = (-1 + 2 * (double)rand()/RAND_MAX);
        }
    }
    return mat;
}
// Fixa memory free bullshit grejer
double twoLayerPerceptron(double learningRate, int numOfHidden, int epochs, Matrix X, Matrix targets)
{
    Matrix W1 = generateRandomMatrix(numOfHidden, X.rows);
    Matrix W2 = generateRandomMatrix(1, numOfHidden+1);
    for (int i = 0; i < epochs; i++)
    {
        Matrix hidden = addRowWithOnes(applyFunction(matrixMultiplication(W1, X), phi));
        Matrix out = applyFunction(matrixMultiplication(W2, hidden), phi);

        Matrix diff = matrixSubtraction(out, targets);
        Matrix o_deriv = applyFunction(out, phiDeriv);
        Matrix delta_o = elementwiseMultiplication(diff, o_deriv);
        freeMatrix(diff);
        freeMatrix(o_deriv);

        Matrix b = matrixMultiplication(transpose(W2), delta_o);
        Matrix h_deriv = applyFunction(hidden, phiDeriv);
        Matrix delta_h = elementwiseMultiplication(b, h_deriv);
        freeMatrix(b);
        freeMatrix(h_deriv);
        
        delta_h = removeRow(delta_h, delta_h.rows - 1);
        Matrix gradient_w1 = matrixMultiplication(delta_h, transpose(X));
        
        Matrix gradient_w2 = matrixMultiplication(delta_o, transpose(hidden));

        freeMatrix(delta_h);
        freeMatrix(delta_o);
        freeMatrix(hidden);

        Matrix delta_w1 = negativeMatrix(gradient_w1);
        Matrix delta_w2 = negativeMatrix(gradient_w2);

        freeMatrix(gradient_w1);
        freeMatrix(gradient_w2);

        Matrix delta_w1_lr = scalarMultiplication(delta_w1, learningRate);
        Matrix delta_w2_lr = scalarMultiplication(delta_w2, learningRate);
        freeMatrix(delta_w1);
        freeMatrix(delta_w2);

        W1 = matrixAddition(W1, delta_w1_lr);
        W2 = matrixAddition(W2, delta_w2_lr);
    }
    //grejer här som saknas

    return 0;
}

int main(int argc, char const *argv[])
{
    char *filepath = "../data/1dFuncData.txt";
    Matrix input = read1DFuncData(filepath);
    filepath = "../data/cringe.txt";
    Matrix input2 = read1DFuncData(filepath);

    twoLayerPerceptron(0.01, 20, 2, input, input2);

    freeMatrix(input);

    return 0;
}
