#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0

#define K 3
#define M 4
#define L 5

int main(int argc, char *argv[])
{
    int id, p;
    float A[K][M];
    float B[M][L];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (id == MASTER)
    {
        float val = 0.0;
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < M; j++)
            {
                A[i][j] = val++;
            }
        }
        val = 0.0;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < L; j++)
            {
                B[i][j] = val++;
            }
        }

        printf("\n=== INITIAL MATRIX A (%dx%d) ===\n", K, M);
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < M; j++)
            {
                printf("%.1f ", A[i][j]);
            }
            printf("\n");
        }

        printf("\n=== INITIAL MATRIX B (%dx%d) ===\n", M, L);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < L; j++)
            {
                printf("%.1f ", B[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    int localM = M / p; // Proveriti ovo

    MPI_Datatype block_A, block_B, block_A_resized, block_B_resized;

    MPI_Type_vector(K, localM, M, MPI_FLOAT, &block_A);
    MPI_Type_create_resized(block_A, 0, localM * sizeof(float), &block_A_resized);
    MPI_Type_commit(&block_A_resized);

    MPI_Type_contiguous(localM * L, MPI_FLOAT, &block_B);
    MPI_Type_create_resized(block_B, 0, localM * L * sizeof(float), &block_B_resized);
    MPI_Type_commit(&block_B_resized);

    float *localA = (float *)malloc(K * localM * sizeof(float));
    float *localB = (float *)malloc(localM * L * sizeof(float));

    struct
    {
        float val;
        int id;
    } localAmin, globalAmin;

    float *localC = (float *)malloc(K * L * sizeof(float));

    MPI_Scatter(A, 1, block_A_resized, localA, K * localM, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(B, 1, block_B_resized, localB, localM * L, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    localAmin.val = localA[0];
    localAmin.id = id;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < L; j++)
        {
            localC[i * L + j] = 0.0;
        }

    // Check this out
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < L; j++)
        {
            for (int k = 0; k < localM; k++)
            {
                localC[i * L + j] += localA[i * localM + k] * localB[k * L + j];
                if (localA[i * localM + k] < localAmin.val)
                {
                    localAmin.val = localA[i * localM + k];
                    localAmin.id = id;
                }
            }
        }
    }

    float *C = (float *)malloc(K * L * sizeof(float));

    MPI_Reduce(localC, C, K * L, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&localAmin, &globalAmin, 1, MPI_FLOAT_INT, MPI_MINLOC, MASTER, MPI_COMM_WORLD);

    MPI_Bcast(&globalAmin, 1, MPI_FLOAT_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(C, K * L, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    if (globalAmin.id == id)
    {
        printf("\n=== RESULTING MATRIX C = A x B (%dx%d) ===\n", K, L);
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < L; j++)
            {
                printf("%.1f ", C[i * L + j]);
            }
            printf("\n");
        }
        printf("\n=== MINIMUM VALUE IN MATRIX A ===\n");
        printf("Process %d has the minimum value: %.1f\n", id, globalAmin.val);
        printf("\n");
    }

    free(localA);
    free(localB);
    free(localC);
    free(C);
    MPI_Type_free(&block_A);
    MPI_Type_free(&block_B);
    MPI_Type_free(&block_A_resized);
    MPI_Type_free(&block_B_resized);
    MPI_Finalize();

    return 0;
}