#include <stdio.h>
#include "mpi.h"

#define ROOT 0

int main(int argc, char *argv[])
{
    int MyRank, Numprocs;
    int value, sum = 0;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
    if (MyRank == ROOT)
    {
        value = 1;
        MPI_Send(&value, 1, MPI_INT, MyRank + 1, 0,
                 MPI_COMM_WORLD);
    }
    else
    {
        if (MyRank < Numprocs - 1)
        {
            MPI_Recv(&value, 1, MPI_INT, MyRank - 1, 0,
                     MPI_COMM_WORLD, &status);
            sum = MyRank + 1 + value;
            MPI_Send(&sum, 1, MPI_INT, MyRank + 1, 0,
                     MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&value, 1, MPI_INT, MyRank - 1, 0,
                     MPI_COMM_WORLD, &status);
            sum = MyRank + 1 + value;
            printf("MyRank %d Final SUM %d\n", MyRank, sum);
        }
    }
    MPI_Finalize();
    return 0;
}