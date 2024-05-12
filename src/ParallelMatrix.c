#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define ROWS_A 3 // Number of rows in input matrices
#define COLS_A 5  // Number of columns in input matrices
#define ROWS_B COLS_A
#define COLS_B 7

void initialize_matrix(double *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = (double)rand() / RAND_MAX; // Random value between 0 and 1
        }
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    int dims[2], periods[2], coords[2];
    int reorder = 1;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate grid dimensions
    int sqrt_p = sqrt(size);
    dims[0] = sqrt_p;
    dims[1] = size / sqrt_p;
    periods[0] = periods[1] = 1;

    // Create a 2D cartesian grid communicator
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Calculate block sizes
    int rows_per_block = ROWS_A / dims[0] + abs(coords[0] < ROWS_A % dims[0]);
    int cols_per_block = COLS_A / dims[1] + abs(coords[1] < COLS_A % dims[1]);

    // Allocate memory for local matrix blocks
    double *A_block = (double *)malloc(rows_per_block * cols_per_block * sizeof(double));
    double *B_block = (double *)malloc(rows_per_block * cols_per_block * sizeof(double));
    double *C_block = (double *)calloc(rows_per_block * cols_per_block, sizeof(double)); // Initialize to zero

    // Seed random number generator
    srand(time(NULL) + rank);

    // Initialize matrices A and B with random values
    initialize_matrix(A_block, rows_per_block, cols_per_block);
    initialize_matrix(B_block, rows_per_block, cols_per_block);

    // // Cannon's algorithm
    // for (int step = 0; step < dims[0]; step++) {
    //     // Shift A blocks left
    //     // Shift B blocks up

    //     // Local matrix multiplication
    //     for (i = 0; i < rows_per_block; i++) {
    //         for (j = 0; j < cols_per_block; j++) {
    //             for (k = 0; k < COLS; k++) {
    //                 C_block[i * cols_per_block + j] += A_block[i * COLS + k] * B_block[k * cols_per_block + j];
    //             }
    //             printf("%f", C_block[i * cols_per_block + j]);
    //         }
    //     }

    //     // Communication along rows and columns
    // }

    // Cannon's algorithm
    MPI_Status status;
    int left_neighbor, right_neighbor, up_neighbor, down_neighbor;
    MPI_Cart_shift(grid_comm, 1, -1, &right_neighbor, &left_neighbor);
    MPI_Cart_shift(grid_comm, 0, -1, &down_neighbor, &up_neighbor);

    int recv_A_coords[2];
    int recv_B_coords[2];
    MPI_Cart_coords(grid_comm, right_neighbor, 2, recv_A_coords);
    MPI_Cart_coords(grid_comm, down_neighbor, 2, recv_A_coords);
    int recv_A_Size = (ROWS_A / dims[0] + abs(recv_A_coords[0] < ROWS_A % dims[0])) * (COLS_A / dims[1] + abs(recv_A_coords[1] < COLS_A % dims[1]));
    int recv_B_Size = (ROWS_A / dims[0] + abs(recv_B_coords[0] < ROWS_A % dims[0])) * (COLS_A / dims[1] + abs(recv_B_coords[1] < COLS_A % dims[1]));
    double *send_buf_A = (double *)malloc(rows_per_block * cols_per_block * sizeof(double));
    double *recv_buf_A = (double *)malloc(recv_A_Size * sizeof(double));
    double *send_buf_B = (double *)malloc(rows_per_block * cols_per_block * sizeof(double));
    double *recv_buf_B = (double *)malloc(recv_B_Size * sizeof(double));

    for (int i = 0; i < coords[0]; i++)
    {
        if (coords[1] % 2 == 0)
        {
            // send
            MPI_Send(A_block, rows_per_block * cols_per_block, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD);
            // recv
            MPI_Probe(right_neighbor, 0, MPI_COMM_WORLD, &status);
            int received_message_size;
            MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
            double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
            MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            free(A_block);
            A_block = receive_buffer;
        }
        else
        {
            // recv
            MPI_Probe(right_neighbor, 0, MPI_COMM_WORLD, &status);
            int received_message_size;
            MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
            double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
            MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // send
            MPI_Send(A_block, rows_per_block * cols_per_block, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD);
            free(A_block);
            A_block = receive_buffer;
        }
    }

    for (int i = 0; i < coords[1]; i++)
    {
        if (coords[0] % 2 == 0)
        {
            // send
            MPI_Send(B_block, rows_per_block * cols_per_block, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD);
            // recv
            MPI_Probe(down_neighbor, 0, MPI_COMM_WORLD, &status);
            int received_message_size;
            MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
            double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
            MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            free(B_block);
            B_block = receive_buffer;
        }
        else
        {
            // recv
            MPI_Probe(down_neighbor, 0, MPI_COMM_WORLD, &status);
            int received_message_size;
            MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
            double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
            MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // send
            MPI_Send(B_block, rows_per_block * cols_per_block, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD);
            free(B_block);
            B_block = receive_buffer;
        }
    }

    
    // Allocate separate send and receive buffers

    for (int step = 0; step < dims[0]; step++)
    {
        // Local matrix multiplication
        for (i = 0; i < rows_per_block; i++)
        {
            for (j = 0; j < cols_per_block; j++)
            {
                for (k = 0; k < cols_per_block; k++)
                {
                    C_block[i * cols_per_block + j] += A_block[i * cols_per_block + k] * B_block[k * cols_per_block + j];
                }
            }
        }

        if (step != dims[0] - 1)
        {
            if (coords[0] % 2 == 0)
            {
                // send
                MPI_Send(B_block, rows_per_block * cols_per_block, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD);
                // recv
                MPI_Probe(down_neighbor, 0, MPI_COMM_WORLD, &status);
                int received_message_size;
                MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
                double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
                MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                free(B_block);
                B_block = receive_buffer;
            }
            else
            {
                // recv
                MPI_Probe(down_neighbor, 0, MPI_COMM_WORLD, &status);
                int received_message_size;
                MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
                double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
                MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // send
                MPI_Send(B_block, rows_per_block * cols_per_block, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD);
                free(B_block);
                B_block = receive_buffer;
            }

            if (coords[1] % 2 == 0)
            {
                // send
                MPI_Send(A_block, rows_per_block * cols_per_block, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD);
                // recv
                MPI_Probe(right_neighbor, 0, MPI_COMM_WORLD, &status);
                int received_message_size;
                MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
                double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
                MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                free(A_block);
                A_block = receive_buffer;
            }
            else
            {
                // recv
                MPI_Probe(right_neighbor, 0, MPI_COMM_WORLD, &status);
                int received_message_size;
                MPI_Get_count(&status, MPI_DOUBLE, &received_message_size);
                double *receive_buffer = (double *)malloc(received_message_size * sizeof(double));
                MPI_Recv(receive_buffer, received_message_size, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // send
                MPI_Send(A_block, rows_per_block * cols_per_block, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD);
                free(A_block);
                A_block = receive_buffer;
            }
        }
    }
    free(send_buf_A);
    free(recv_buf_A);
    free(send_buf_B);
    free(recv_buf_B);

    

    // Gather results to process 0
    double *C_result = NULL;
    if (rank == 0)
    {
        C_result = (double *)malloc(ROWS_A * COLS_A * sizeof(double));
    }
    //MPI_Gather(C_block, rows_per_block * cols_per_block, MPI_DOUBLE, C_result, rows_per_block * cols_per_block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (i = 0; i < rows_per_block; i++)
    {
        for (j = 0; j < cols_per_block; j++)
        {
            printf("%d: %.2f ",rank, C_block[i*cols_per_block + j]);
        }
    }
    // Finalize
    free(A_block);
    free(B_block);
    free(C_block);
    if (rank == 0)
    {
        free(C_result);
    }
    MPI_Finalize();
    return 0;
}
