#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0
#define N 6

int main(int argc, char **argv)
{
    int myrank, numprocs, i, n_bar;
    float a[N], b[N], dot, local_dot;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    n_bar = N / numprocs;

    for (i = 0; i < N; i++)
    {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 1);
    }

    float *local_a = (float *)malloc(n_bar * sizeof(float));
    float *local_b = (float *)malloc(n_bar * sizeof(float));

    MPI_Scatter(a, n_bar, MPI_FLOAT, local_a, n_bar, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(b, n_bar, MPI_FLOAT, local_b, n_bar, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    for (i = 0; i < n_bar; i++)
    {
        local_dot += local_a[i] * local_b[i];
    }

    MPI_Reduce(&local_dot, &dot, 1, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (myrank == ROOT)
    {
        printf("Dot product is %f\n", dot);
    }

    MPI_Finalize();
    return 0;
}