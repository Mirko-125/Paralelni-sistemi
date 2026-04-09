#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define TOTAL_ELEMENTS 8
#define ROOT 0

int main(int argc, char *argv[])
{
    int data[TOTAL_ELEMENTS];
    MPI_Status status;
    int rank, num_procs, step, elements_to_send, ilevel, hypercube_distance, next_power, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Only root initializes data
    if (rank == ROOT)
        for (i = 0; i < TOTAL_ELEMENTS; i++)
            data[i] = i;

    int num_levels = (int)(log((double)num_procs) / log(2.0)); // log2(p)
    elements_to_send = TOTAL_ELEMENTS;

    for (step = num_levels - 1; step >= 0; step--)
    {
        hypercube_distance = (int)pow(2.0, (double)step); // razmak između sender/receiver
        elements_to_send = elements_to_send / 2;          // svaki korak polovina ide dalje
        next_power = (int)pow(2.0, (double)(step + 1));   // period za određivanje sender/receiver

        if (rank % hypercube_distance == 0) // samo procesi na ovom nivou učestvuju
        {
            if (rank % next_power == 0)
                // Sender: šalje gornju polovinu svog niza susjedu
                MPI_Send(data + elements_to_send, elements_to_send, MPI_INT,
                         rank + hypercube_distance, 0, MPI_COMM_WORLD);
            else
                // Receiver: prima podatke od svog para
                MPI_Recv(data, elements_to_send, MPI_INT,
                         rank - hypercube_distance, 0, MPI_COMM_WORLD, &status);
        }
    }

    for (i = 0; i < elements_to_send; i++)
        printf("Proces %d ima data[%d] = %d\n", rank, i, data[i]);

    MPI_Finalize();
    return 0;
}