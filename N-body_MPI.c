#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "mpi.h"
#include<assert.h>

#define Nparticles 200
#define Dimension 3
#define delta_t 0.1
#define Nsteps 100

// Function Prototype
double *get_rand_num(int num_elements, double* any_array);

double *rand_positions(int num_elements, double* any_array);

int main(int argc, char **argv)
{
    int size;                       /* Size of communicator*/
    int rank;                       /* Rank of processes*/
    double start_time;              /* start time*/
    double end_time;                /* end time */
    int chunk_size;                 /* chunk size for each process*/

    double *masses = NULL;          /* Pointer for Masses*/

    double *position_x = NULL;      /* Pointer for Positions*/
    double *position_y = NULL;
    double *position_z = NULL;

    double *velocity_x = NULL;      /* Pointer for Velocities*/
    double *velocity_y = NULL;
    double *velocity_z = NULL;

    double *sub_velocity_x = NULL;  /* Pointer for Portion of Velocities for each processes*/
    double *sub_velocity_y = NULL;
    double *sub_velocity_z = NULL;

    double acceleration_x[chunk_size];  /* Array for accelerations*/
    double acceleration_y[chunk_size];
    double acceleration_z[chunk_size];

    double *new_position_x = NULL;  /* Pointer for new positions*/
    double *new_position_y = NULL;
    double *new_position_z = NULL;

    double *final_position_x = NULL;    /* Pointer for final position storage*/
    double *final_position_y = NULL;
    double *final_position_z = NULL;

    /* Initialize MPI Environment */
    MPI_Init(&argc, &argv);

    /* Get size of Processes in MPI Environment*/
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Get rank of each processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    chunk_size = Nparticles / size;
    masses = (double *) malloc(sizeof(double) * Nparticles);
    assert(masses != NULL);
    position_x = (double *) malloc(sizeof(double) * Nparticles);
    assert(position_x != NULL);
    position_y = (double *) malloc(sizeof(double) * Nparticles);
    assert(position_y != NULL);
    position_z = (double *) malloc(sizeof(double) * Nparticles);
    assert(position_z != NULL);
    sub_velocity_x = (double*) malloc(sizeof(double)*chunk_size);
    assert(sub_velocity_x != NULL);
    sub_velocity_y = (double*) malloc(sizeof(double)*chunk_size);
    assert(sub_velocity_y != NULL);
    sub_velocity_z = (double*) malloc(sizeof(double)*chunk_size);
    assert(sub_velocity_z != NULL);

    new_position_x = (double*) malloc(sizeof(double)* chunk_size);
    assert(new_position_x != NULL);
    new_position_y = (double*) malloc(sizeof(double)* chunk_size);
    assert(new_position_y != NULL);
    new_position_z = (double*) malloc(sizeof(double)* chunk_size);
    assert(new_position_z != NULL);


    if (rank == 0)
    {
        velocity_x = (double *) malloc(sizeof(double) * Nparticles);
        assert(velocity_x != NULL);
        velocity_y = (double *) malloc(sizeof(double) * Nparticles);
        assert(velocity_y != NULL);
        velocity_z = (double *) malloc(sizeof(double) * Nparticles);
        assert(velocity_z != NULL);

        // Get Random Masses of particles
        masses = get_rand_num(Nparticles, masses);
        position_x = rand_positions(Nparticles, position_x);
        position_y = rand_positions(Nparticles, position_y);
        position_z = rand_positions(Nparticles, position_z);
        velocity_x = rand_positions(Nparticles, velocity_x);
        velocity_y = rand_positions(Nparticles, velocity_y);
        velocity_z = rand_positions(Nparticles, velocity_z);

        final_position_x = (double *) malloc(sizeof(double) * Nparticles);
        assert(final_position_x != NULL);
        final_position_y = (double *) malloc(sizeof(double) * Nparticles);
        assert(final_position_y != NULL);
        final_position_z = (double *) malloc(sizeof(double) * Nparticles);
        assert(final_position_z != NULL);
    }
    start_time = MPI_Wtime();
    /* MPI_Bcast(void* data, int count, MPI_Datatype datatype,
                int root, MPI_Comm communicator)*/
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(masses, Nparticles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(position_x, Nparticles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(position_y, Nparticles, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(position_z, Nparticles, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* MPI Scatter to send all arrays to other processes*/
    /* MPI_Scatter(void* send_data, int send_count, MPI_Datatype send_datatype,
        void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator) */
    MPI_Scatter(velocity_x, chunk_size, MPI_DOUBLE, sub_velocity_x, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(velocity_y, chunk_size, MPI_DOUBLE, sub_velocity_y, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(velocity_z, chunk_size, MPI_DOUBLE, sub_velocity_z, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // MPI_Barrier(MPI_COMM_WORLD);
    for(int i=1; i<=Nsteps; i++)
    {
        for(int j=0; j<chunk_size; j++)
        {
            for(int m=0; m<chunk_size; m++)
            {
                acceleration_x[m] = 0.0;
                acceleration_y[m] = 0.0;
                acceleration_z[m] = 0.0;
            }
            int check_particle = rank*chunk_size + j;
            for(int k=0; k<Nparticles; k++)
            {
                double delta_x, delta_y, delta_z;
                double temp, invr, invr3, force;
                double eps;
                if (k != check_particle)
                {
                    delta_x = position_x[check_particle] - position_x[k];
                    delta_y = position_y[check_particle] - position_y[k];
                    delta_z = position_z[check_particle] - position_z[k];
                    temp = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + eps);
                    invr = 1.0 / temp;
                    invr3 = invr*invr*invr;
                    force = masses[k] * invr3;
                    acceleration_x[k] += force*delta_x;
                    acceleration_y[k] += force*delta_y;
                    acceleration_z[k] += force*delta_z;
                }
            }
            new_position_x[j] = position_x[check_particle] + delta_t*sub_velocity_x[j] + 0.5*delta_t*delta_t*acceleration_x[j];
            new_position_y[j] = position_y[check_particle] + delta_t*sub_velocity_y[j] + 0.5*delta_t*delta_t*acceleration_y[j];
            new_position_z[j] = position_z[check_particle] + delta_t*sub_velocity_z[j] + 0.5*delta_t*delta_t*acceleration_z[j];
            sub_velocity_x[j] += delta_t*acceleration_x[j];
            sub_velocity_y[j] += delta_t*acceleration_y[j];
            sub_velocity_z[j] += delta_t*acceleration_z[j];
            position_x[rank*chunk_size + j] = new_position_x[j];
            position_y[rank*chunk_size + j] = new_position_y[j];
            position_z[rank*chunk_size + j] = new_position_z[j];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        /* MPI_Gather(void* send_data, int send_count, MPI_Datatype send_datatype,
                void* recv_data, int recv_count, MPI_Datatype recv_datatype,
                int root, MPI_Comm communicator)*/
        MPI_Gather(new_position_x, chunk_size, MPI_DOUBLE, final_position_x, chunk_size, MPI_DOUBLE, 0,MPI_COMM_WORLD);
        MPI_Gather(new_position_y, chunk_size, MPI_DOUBLE, final_position_y, chunk_size, MPI_DOUBLE, 0,MPI_COMM_WORLD);
        MPI_Gather(new_position_z, chunk_size, MPI_DOUBLE, final_position_z, chunk_size, MPI_DOUBLE, 0,MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (i == 1)
            {
                FILE *output_file = fopen ( "initial_position.csv","w+");
                fprintf(output_file, "X,Y,Z \n");
                for(int i=0; i<Nparticles; i++)
                {
                    fprintf(output_file, "%lf, %lf, %lf \n", final_position_x[i], final_position_y[i], final_position_z[i]);
                }
                fclose(output_file);

            }

            if (i == Nsteps)
            {
                FILE *output_file = fopen ( "final_position.csv","w+");
                fprintf(output_file, "X,Y,Z \n");
                for(int i=0; i<Nparticles; i++)
                {
                    fprintf(output_file, "%lf, %lf, %lf \n", final_position_x[i], final_position_y[i], final_position_z[i]);
                }
                fclose(output_file);

            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank == 0)
    {
        printf("********************* Stats of Simulation *********************\n");
        printf("\t Number of MPI Processees = %d \t \n", size);
        printf("\t Number of Particles = %d \t \n", Nparticles);
        printf("\t Total Time = %lf \t \n", Nsteps*delta_t);
        printf("\t Time step = %lf \t \n", delta_t);
        printf("\t Total Steps = %d\t \n", Nsteps);
        printf("\t Time Elapsed = %e \t \n", end_time-start_time);
        printf("***************************************************************\n");

    }
    free(masses);
    free(position_x);
    free(position_y);
    free(position_z);

    free(sub_velocity_x);
    free(sub_velocity_y);
    free(sub_velocity_z);

    free(new_position_x);
    free(new_position_y);
    free(new_position_z);

    if (rank == 0)
    {
        free(velocity_x);
        free(velocity_y);
        free(velocity_z);
        free(final_position_x);
        free(final_position_y);
        free(final_position_z);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}


/* Generate Random number of Masses according to number of particles*/
double *get_rand_num(int num_elements, double* any_array)
{
    int i;
    for(i=0; i<num_elements; i++)
    {
        any_array[i] = rand() % 10 + 1;
    }
    return any_array;
}

/* Generate random number of Positions */
double *rand_positions(int num_elements, double* any_array)
{
    int i;
    for(i=0; i<num_elements; i++)
    {
        any_array[i] = rand() % 100;
    }
    return any_array;
}