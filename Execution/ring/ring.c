#include "mpi.h"
#include <stdio.h>

int main(int argc, char **argv)
{ 
	int rank, np, message;
	MPI_Status recv_status, send_status;
	int next, last, in;
	MPI_Request request;
	int tag = 23; // arbitrary value

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// work out identity of neighbours
	next = (rank + 1) % np;
	last = (rank + np - 1) % np;
	in = -1;
	message = rank;
	
	// messages circle until each process receives its own id back again
	while (in != rank) 
	{
		MPI_Isend(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD, &request);
		MPI_Recv(&in, 1, MPI_INT, last, tag, MPI_COMM_WORLD, &recv_status); 
		printf("Process %d received %d\n", rank, in);
		MPI_Wait(&request, &send_status);
		message = in;
	}
	MPI_Finalize();
}


