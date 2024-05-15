#include <time.h>
#include "Parallel.c"

int main(int argc, char *argv[])
{
    int rank, size, rc;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix input;
    Matrix targets;
    if (argc == 2)
    {
        if (rank == 0)
        {
            char *filepath = "data/input.txt";
            input = read1DFuncData(filepath);
            filepath = "data/output.txt";
            targets = read1DFuncData(filepath);
        }
    }
    else if (argc == 4)
    {
        // random matrices for mlp
        if (rank == 0)
        {
            input = generateRandomMatrix(atoi(argv[2]), atoi(argv[3]));
            targets = generateRandomMatrix(1, atoi(argv[3]));
        }
    }
    else if (argc == 5)
    {
        // random matrix MxN * NxK
        if (rank == 0)
        {
            input = generateRandomMatrix(atoi(argv[2]), atoi(argv[3]));
            targets = generateRandomMatrix(atoi(argv[3]), atoi(argv[4]));
        }
        if (*argv[1] == 'c')
        {
            // cannons only

            clock_t start = clock();
            parallelMatrixMult(input, targets, rank, size);
            clock_t difference = clock() - start;
            if(rank == 0){
                printf("Time: %f\n", difference / (float)CLOCKS_PER_SEC);
                printf("%sx%s times %sx%s\n", argv[2], argv[3], argv[3], argv[4]);
            }
        }
        else if (*argv[1] == 'n')
        {
            // naive only
            if (rank == 0)
            {
                clock_t start = clock();
                matrixMultiplication(input, targets);
                clock_t difference = clock() - start;
                printf("Time: %f\n", difference / (float)CLOCKS_PER_SEC);
                printf("%sx%s times %sx%s\n", argv[2], argv[3], argv[3], argv[4]);
            }
            
        }
        else
        {
            if (rank == 0)
            {
                printf("first argument must be c or n\n");
            }
        }
        MPI_Finalize();
        return 0;
    }
    else
    {
        if (rank == 0)
        {
            printf("Invalid arguments\n");
        }
        MPI_Finalize();
        return 0;
    }

    double learningRate = 0.001;
    int numOfHidden = 25;
    int epochs = 10;

    if (*argv[1] == 's')
    {
        if (rank == 0)
        {
            clock_t start = clock();
            double mse = twoLayerPerceptron(learningRate, numOfHidden, epochs, input, targets);
            clock_t difference = clock() - start;

            printf("Time: %f\n", difference / (float)CLOCKS_PER_SEC);

            printf("MSE: %f\n", mse);
        }
    }
    else if (*argv[1] == 'p')
    {
        // Parallel

        if (rank == 0)
        {
            clock_t start = clock();
            double mse = parallelTwoLayerPerceptron(learningRate, numOfHidden, epochs, input, targets, rank, size);
            clock_t difference = clock() - start;
            printf("Time: %f\n", difference / (float)CLOCKS_PER_SEC);

            printf("MSE: %f\n", mse);
        }
        else
        {
            parallelTwoLayerPerceptron(learningRate, numOfHidden, epochs, input, targets, rank, size);
        }
    }
    else
    {
        if(rank == 0){
            printf("first argument must be s for seqential, or p for parallel\n");
        }
    }
    MPI_Finalize();

    return 0;
}