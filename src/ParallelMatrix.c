#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "Project.c"

void red_black(double** Block, int coord, int known_dim, int* modifiable_dim, int send_neighbor, int recv_neighbor, MPI_Status status){
     
    if (coord % 2 == 0)
    {
        // send
        MPI_Send(*Block, (*modifiable_dim) * known_dim, MPI_DOUBLE, send_neighbor, 0, MPI_COMM_WORLD);
        // recv
        MPI_Probe(recv_neighbor, 0, MPI_COMM_WORLD, &status);
        int received_message_size;
        MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
        double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
        MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, recv_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *modifiable_dim = received_message_size / known_dim;
        free(*Block);
        *Block = receive_buffer;
    }
    else
    {
        // recv
        MPI_Probe(recv_neighbor, 0, MPI_COMM_WORLD, &status);
        int received_message_size;
        MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
        double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
        MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, recv_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // send
        MPI_Send(*Block, known_dim * (*modifiable_dim), MPI_DOUBLE, send_neighbor, 0, MPI_COMM_WORLD);
        *modifiable_dim = received_message_size / known_dim;
        free(*Block);
        *Block = receive_buffer;
    }

}

Matrix parallelMatrixMult(Matrix A, Matrix B, int rank, int size)
{
    int dims[2], periods[2], coords[2];
    int reorder = 1;
    int i, j, k;

    

    // Calculate grid dimensions
    int sqrt_p = sqrt(size);
    
    dims[0] = sqrt_p;
    dims[1] = size / sqrt_p;
    
    periods[0] = periods[1] = 1;

    // Create a 2D cartesian grid communicator
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int A_rows, A_cols, B_rows, B_cols;
    if(rank == 0){
        A_rows = A.rows;
        A_cols = A.cols;
        B_rows = B.rows;
        B_cols = B.cols;
    }
    MPI_Bcast(&A_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate block sizes
    int rows_per_A_block = A_rows / dims[0] + abs(coords[0] < A_rows % dims[0]);
    int cols_per_A_block = A_cols / dims[1] + abs(coords[1] < A_cols % dims[1]);

    int rows_per_B_block = B_rows / dims[1] + abs(coords[0] < B_rows % dims[1]);
    int cols_per_B_block = B_cols / dims[0] + abs(coords[1] < B_cols % dims[0]);

    // Allocate memory for local matrix blocks
    double *A_block = (double *)malloc(rows_per_A_block * cols_per_A_block * sizeof(double));
    double *B_block = (double *)malloc(rows_per_B_block * cols_per_B_block * sizeof(double));
    double *C_block = (double *)calloc(rows_per_A_block * cols_per_B_block, sizeof(double)); // Initialize to zero

    // distribute matrices
    if (rank == 0)
    {
        int current_x = cols_per_A_block;
        int current_y = 0;
        int current_x_B = cols_per_B_block;
        int current_y_B = 0;
        int send_coords[2];
        // send loop
        for (int p = 1; p < size; p++)
        {
            
            MPI_Cart_coords(grid_comm, p, 2, send_coords);
            // send A
            int send_cols_A = A_cols / dims[1] + abs(send_coords[1] < A_cols % dims[1]);
            int send_rows_A = A_rows / dims[0] + abs(send_coords[0] < A_rows % dims[0]);
            
            for (size_t i = 0; i < send_rows_A; i++)
            {
                for (size_t j = 0; j < send_cols_A; j++)
                {
                    A_block[i * send_cols_A + j] = A.data[i + current_y][j + current_x];
                }
            }
            MPI_Send(A_block, send_rows_A * send_cols_A, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            current_x += send_cols_A;
            if (current_x == A_cols)
            {
                current_x = 0;
                current_y += send_rows_A;
            }
            // send B
            int send_cols_B = B_cols / dims[1] + abs(send_coords[1] < B_cols % dims[1]);
            int send_rows_B = B_rows / dims[0] + abs(send_coords[0] < B_rows % dims[0]);
            for (size_t i = 0; i < send_rows_B; i++)
            {
                for (size_t j = 0; j < send_cols_B; j++)
                {
                    B_block[i * send_cols_B + j] = B.data[i + current_y_B][j + current_x_B];
                }
            }
            MPI_Send(B_block, send_rows_B * send_cols_B, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            
            current_x_B += send_cols_B;
            if (current_x_B == B_cols)
            {
                current_x_B = 0;
                current_y_B += send_rows_B;
            }
        }
        // set rank 0 blocks
        for (size_t i = 0; i < rows_per_A_block; i++)
        {
            for (size_t j = 0; j < cols_per_A_block; j++)
            {
                A_block[i * cols_per_A_block + j] = A.data[i][j];
            }
        }
        for (size_t i = 0; i < rows_per_B_block; i++)
        {
            for (size_t j = 0; j < cols_per_B_block; j++)
            {
                B_block[i * cols_per_B_block + j] = B.data[i][j];
            }
        }
    }
    else
    {
        // recv
        MPI_Recv(A_block, rows_per_A_block * cols_per_A_block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_block, rows_per_B_block * cols_per_B_block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
    }
    
    // Cannon's algorithm
    MPI_Status status;
    int left_neighbor, right_neighbor, up_neighbor, down_neighbor;
    MPI_Cart_shift(grid_comm, 1, -1, &right_neighbor, &left_neighbor);
    MPI_Cart_shift(grid_comm, 0, -1, &down_neighbor, &up_neighbor);
    int new_rows_B;
    int new_cols_A;
    // Initial shifts in cannon's algo, depending on row/column index
    for (int i = 0; i < coords[0]; i++)
    {
        red_black(&A_block, coords[1], rows_per_A_block, &cols_per_A_block, left_neighbor, right_neighbor, status);
    }

    for (int i = 0; i < coords[1]; i++)
    {
        red_black(&B_block, coords[0], cols_per_B_block, &rows_per_B_block, up_neighbor, down_neighbor, status);
    }

    for (int step = 0; step < dims[0]; step++)
    {
        // Local matrix multiplication
        for (i = 0; i < rows_per_A_block; i++)
        {
            for (j = 0; j < cols_per_A_block; j++)
            {
                for (k = 0; k < cols_per_B_block; k++)
                {
                    C_block[i * cols_per_B_block + k] += A_block[i * cols_per_A_block + j] * B_block[j * cols_per_B_block + k];
                }
            }
        }
        if (step != dims[0] - 1)
        {
            red_black(&B_block, coords[0], cols_per_B_block, &rows_per_B_block, up_neighbor, down_neighbor, status);   
            red_black(&A_block, coords[1], rows_per_A_block, &cols_per_A_block, left_neighbor, right_neighbor, status);   
        }
    }

    // Gather results to process 0
    Matrix C_result;
    if (rank == 0)
    {
        C_result = createMatrix(A_rows, B_cols);
        int current_x_pos = 0;
        int current_y_pos = 0;

        // recv loop
        for (int i = 0; i < size; i++)
        {

            if (i != 0)
            {
                MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
                int received_message_size;
                MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
                double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
                MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                free(C_block);
                C_block = receive_buffer;
                int recv_coords[2];
                MPI_Cart_coords(grid_comm, i, 2, recv_coords);

                rows_per_A_block = A_rows / dims[0] + abs(recv_coords[0] < A_rows % dims[0]);
                cols_per_B_block = B_cols / dims[0] + abs(recv_coords[1] < B_cols % dims[0]);
            }
            for (size_t i = 0; i < rows_per_A_block; i++)
            {
                for (size_t j = 0; j < cols_per_B_block; j++)
                {
                    C_result.data[current_y_pos + i][j + current_x_pos] = C_block[i * cols_per_B_block + j];
                }
            }
            current_x_pos += cols_per_B_block;
            if (current_x_pos == B_cols)
            {
                current_x_pos = 0;
                current_y_pos += rows_per_A_block;
            }
        }
    }
    else
    {
        // send
        MPI_Send(C_block, rows_per_A_block * cols_per_B_block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    free(A_block);
    free(B_block);
    free(C_block);
    
    if (rank == 0)
    {
        freeMatrix(A);
        freeMatrix(B);
        return C_result;
    }
    
}

/* int main(int argc, char *argv[])
{
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Matrix a;
    if(rank == 0){
        Matrix A = createMatrix(1000,1000);
        for (size_t i = 0; i < A.rows; i++)
        {
            for (size_t j = 0; j < A.cols; j++)
            {
                A.data[i][j] = i*A.cols + j;
            }
        }

        Matrix B = createMatrix(1000,1000);
        for (size_t i = 0; i < B.rows; i++)
        {
            for (size_t j = 0; j < B.cols; j++)
            {
                if (i == j)
                {
                    B.data[i][j] = 1;
                }
                else
                {
                    B.data[i][j] = 0.5f;
                }
            }
        }
        
        clock_t start = clock();
        a = parallelMatrixMult(argc, argv, copy(A), copy(B), rank, size);
        
        clock_t difference = clock() - start;
        printf("time: %f\n", difference/(float)CLOCKS_PER_SEC);
        start = clock();
        Matrix b = matrixMultiplication(A, B);
        difference = clock() - start;
        printf("time: %f", difference/(float)CLOCKS_PER_SEC);

        for (size_t i = 0; i < a.rows; i++)
        {
            for (size_t j = 0; j < a.cols; j++)
            {
                if(a.data[i][j] != b.data[i][j]){
                    printf("a: %f, b: %f",a.data[i][j], b.data[i][j] );
                }
            }
            
        }
        
    }
    else{
        Matrix q;
        Matrix s;
        parallelMatrixMult(argc, argv, q, s, rank, size);
    }

    MPI_Finalize();
    return 0;
} */