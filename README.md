## How to run
Start by compiling the main.c file using the following command:

`mpicc src/main.c -o out -lm`

Once compiled, there are several ways of running the program depending on how many parameters you use.

To run a sequential version of the MLP using the files in the data folder, simply execute `mpirun ./out s` and to run it with the parallel implementation, execute `mpirun -n <NumOfProcessors> ./out p`. NOTE: `<NumOfProcessors>` must be a square number (4, 16, 25, etc)

For running the MLP using randomly generated data of size M x N execute the following command: `mpirun ./out s <M> <N>` for a sequential version, and replacing s with p for the parallel version, also include `-n <NumOfProcessors>`. NOTE: `<NumOfProcessors>` must be a square number (4, 16, 25, etc)

Furthermore the program can be used to perform a single matrix multiplication with randomly generated matrices of specified sizes using the following command: `mpirun ./out n <M> <N> <K>` for the naive algorithm and swapping `n` with `c` and adding `-n <NumOfProcessors>` to use cannon's algorithm. This performs a matrix multiplication of matrices with sizes M x N and N x K. NOTE: `<NumOfProcessors>` must be a square number (4, 16, 25, etc). M and K must be greater than or equal to the square root of `<NumOfProcessors>`