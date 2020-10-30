//***************************************************************************************//
// Game of Life - MPI, OpenMP Implementation in C
// Nikos Lazaridis, M1485
// University of Athens (UoA), Department of Informatics (DIT),
// Parallel Computing Systems semester project
//***************************************************************************************//

//////////////////////////////////////////////////////////////////
//////////////////////////// Includes ////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <stdbool.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

#if defined (_WIN32) || defined (_WIN64)
	#include <Windows.h>// for Sleep()
	#define sleep Sleep
#elif defined (__unix__) || defined (__linux__)
	#include <unistd.h> // for usleep()
	#define sleep(ms) usleep(ms * 1000)
#endif

/////////////////////////////////////////////////////////////////
//////////////////////////// Globals ////////////////////////////
static const int master = 0;	// Master process #0
static int np;					// No of processes
static int rank;				// process rank
// neighbour ranks
static int west, north, south, east, northWest, southWest, northEast, southEast;

static int Rows = 24;			// rows of the entire grid
static int Cols = 48;			// columns of the entire grid
static int rows;				// rows of local array
static int cols;				// columns of local array
static int generations = 1000;
static double density = 0.25;
static int prows;				// rows and columns of blocks
static int pcols;				// Each block is handled by 1 process at a time
static int timeStep = 500;		// timeStep for the simulation (in ms)

#ifdef DISPLAY_RESULTS
	#define updateScreen() printf("\033[H")	// places cursor to Home position (top left)
#endif
#if defined(_DEBUG) || defined(DEBUG)
	#define ASSERT(ret, num) \
		do { \
			if (ret != num) \
			{ \
				fprintf(stderr, "\nError [%d] at line [%d] in function [%s]. Date: [%s] " \
						, ret, __LINE__, __func__, __DATE__); \
				exit(ret); \
			} \
		} while (0)
#else
	#define ASSERT(ret, num) 
#endif

//////////////////////////////////////////////////////////////////
///////////////////// Function Declarations //////////////////////
//extern _Noreturn void err_report(int err, int line, const char *func);

void parseCmd(int* argc, char*** argv);
#ifdef DISPLAY_RESULTS
	void display2dGrid(char** grid, int nrows, int ncolumns, int turn);
#endif
#if defined(_DEBUG) || defined(DEBUG)
	void writeGridToFile(char** grid, int nrows, int ncolumns);
#endif
char** allocContiguous2dArr(char* gridData, int nrows, int ncolumns);
void deallocContiguous2dArr(char** grid, char* gridData);
char** randomlyPopulateGrid(char** grid, int nrows, int ncolumns);
int countNeighbors(char** grid, int y, int x);
char** evolve(char** grid, int nrows, int ncolumns);
void determineNeighbourRanks();

void help() 
{
	if (rank == 0) 
	{
		printf("run with:\n mpiexec -n [np] ./[program_name] [rows] [columns] " \
				"[generations] [density (0-1)]\n");
		printf("Processes should be a multiple of (9 + 3*k, k = unsigned int) for optimal" 
				"performance\n");
	}
}


//////////////////////////////////////////////////////////////////
////////////////////////////   main   ////////////////////////////
int main(int argc, char** argv) 
{
	// I. Initialize Environment	//
	int errNo = MPI_SUCCESS;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#if defined(_DEBUG) || defined(DEBUG)
	char version[200];
	char processorName[20];
	int strLen;
	MPI_Get_library_version(version, &strLen);
	MPI_Get_processor_name(processorName, &strLen);
	printf("\n\n%s\n\n", version);
	printf("Processor name: %s\n", processorName);
#endif

	/********************************************************************************/
	// II. Parse command line arguments	//
	parseCmd(&argc, &argv);

	/********************************************************************************/
	// III. Create the Arrays and populate them	//
	char* gridData = NULL;	// needed to create contiguous 2d array
	char** grid = allocContiguous2dArr(gridData, rows, cols);	// local array
	char** newGrid = NULL;										// local array for next gen
#ifdef DISPLAY_RESULTS
	char* universeData = NULL;		// necessary to create contiguous global array
	char** universe = NULL;			// Global array
	char* recvPtr = NULL;			// Pointer to start of Global array
	int ndims = 2;					// array dimensions
	int nelems[2];					// No of elements in each dimension of the entire array
	int nsubElems[2];				// No of elements in each dimension of the subarray
	int startCoords[2];				// Starting coordinates of each subarray in the entire grid
	MPI_Datatype pre_recvBlock, recvBlock;	// receiver's block types
	MPI_Datatype sendBlock;					// sender's block type

	// A. Create receiver data type	//
	if (rank == master) 
	{
		// allocate space for the entire grid
		universe = allocContiguous2dArr(universeData, Rows, Cols);
		recvPtr = &universe[0][0];	// Only the root process knows the memLoc of the global array
		
		// Create a subarray (a rectangular block) datatype from a regular, 2d array
		nelems[0] = Rows;
		nelems[1] = Cols;
		nsubElems[0] = rows - 2;
		nsubElems[1] = cols - 2;
		startCoords[0] = 0;
		startCoords[1] = 0;

		errNo =	MPI_Type_create_subarray(ndims, nelems, nsubElems, startCoords, MPI_ORDER_C, MPI_CHAR, &pre_recvBlock);
		ASSERT(errNo, MPI_SUCCESS);

		// we resize the datatype, by specifying its extent (width). By doing so, we are saying 
		// that the global array will recognize the new type (block) every extent Bytes and change 
		// to the appropriate process / block to acquire the data
		// the extent is (columns - 2) * sizeof(char) = (columns - 2)
		errNo =
		MPI_Type_create_resized(pre_recvBlock, 0, (cols - 2) * sizeof(char), &recvBlock);
		ASSERT(errNo, MPI_SUCCESS);
		errNo =
		MPI_Type_commit(&recvBlock);
		ASSERT(errNo, MPI_SUCCESS);
	}

	/****************************************/
	// B. Create sender data type //
	// we want to specify that we DO want to send the block with the ghost cells
	// So we cut out the ghost rows and columns
	nelems[0] = rows;
	nelems[1] = cols;
	nsubElems[0] = rows - 2;
	nsubElems[1] = cols - 2;
	startCoords[0] = 1;
	startCoords[1] = 1;
	
	errNo = MPI_Type_create_subarray(ndims, nelems, nsubElems, startCoords, MPI_ORDER_C, MPI_CHAR, &sendBlock);
	ASSERT(errNo, MPI_SUCCESS);
	MPI_Type_commit(&sendBlock);
#endif

	// We need a different random seed for EACH process
	grid = randomlyPopulateGrid(grid, rows, cols);

	/********************************************************************************/	
	// IV. Determine neighbouring Processes	//
	determineNeighbourRanks();

	/********************************************************************************/
	// V. Main Game Loop	//
	int turn = 0;
	int i;
#ifdef DISPLAY_RESULTS
	int recvBlocks[np];
	int displacements[np];
	int prow, pcol;
#endif

	char sendLeftColumn[rows];
	char sendRightColumn[rows];
	char recvLeftColumn[rows];
	char recvRightColumn[rows];
	MPI_Request sendRequests[8], 
				recvRequests[8];
	MPI_Status  sendStatuses[8], 
				recvStatuses[8];
	//MPI_Status sendRecvStatuses[8];

	MPI_Barrier(MPI_COMM_WORLD);
	double startTime = MPI_Wtime();

	while (turn < generations) 
	{
		// A. Communication	//
		#pragma omp parallel for shared(rows, cols, grid, sendLeftColumn, sendRightColumn) private(i) num_threads(4)
		for (i = 1; i < rows - 1; i++) 
		{
			sendLeftColumn[i] = grid[i][1];
			sendRightColumn[i] = grid[i][cols-2];
		}

		if (np >= 9) 
		{
			// An optimization step has been employed. By "preparing" communication requests to 
			// avoid overhead later on from argument passing on functions etc.
			// 1. Send requests
			MPI_Send_init(grid[1], cols, MPI_CHAR, north, 0,				// send top row
					MPI_COMM_WORLD, &sendRequests[0]);
			MPI_Send_init(grid[rows-2], cols, MPI_CHAR, south, 0,			// send bottom row
					MPI_COMM_WORLD, &sendRequests[1]);
			MPI_Send_init(sendLeftColumn, rows, MPI_CHAR, west, 0,			// send left column
					MPI_COMM_WORLD, &sendRequests[2]);
			MPI_Send_init(sendRightColumn, rows, MPI_CHAR, east, 0,			// send right column
					MPI_COMM_WORLD, &sendRequests[3]);
			// Prepare ghost cells for sending
			MPI_Send_init(&grid[1][1], 1, MPI_CHAR, northWest, 0,			// send NW cell
					MPI_COMM_WORLD, &sendRequests[4]);
			MPI_Send_init(&grid[1][cols-2], 1, MPI_CHAR, northEast, 0,		// send NE cell
					MPI_COMM_WORLD, &sendRequests[5]);
			MPI_Send_init(&grid[rows-2][1], 1, MPI_CHAR, southWest, 0,		// send SW cell
					MPI_COMM_WORLD, &sendRequests[6]);
			MPI_Send_init(&grid[rows-2][cols-2], 1, MPI_CHAR, southEast, 0,	// send SE cell
					MPI_COMM_WORLD, &sendRequests[7]);
			// Initiate all send requests - wait for them to finish
			MPI_Startall(8, sendRequests);
			MPI_Waitall(8, sendRequests, sendStatuses);

			// 2. Receive requests
			MPI_Recv_init(grid[rows-1], cols, MPI_CHAR, south, 0,			// receive south row
					MPI_COMM_WORLD, &recvRequests[0]);
			MPI_Recv_init(grid[0], cols, MPI_CHAR, north, 0,				// receive north row
					MPI_COMM_WORLD, &recvRequests[1]);
			MPI_Recv_init(recvRightColumn, rows, MPI_CHAR, east, 0,			// receive right col
					MPI_COMM_WORLD, &recvRequests[2]);
			MPI_Recv_init(recvLeftColumn, rows, MPI_CHAR, west, 0,			// receive left col
					MPI_COMM_WORLD, &recvRequests[3]);
			// prepare ghost cells for receiving
			MPI_Recv_init(&grid[rows-1][cols-1], 1, MPI_CHAR, southEast, 0,	// receive SE cell
					MPI_COMM_WORLD, &recvRequests[4]);
			MPI_Recv_init(&grid[rows-1][0], 1, MPI_CHAR, southWest, 0, 		// receive SW cell
					MPI_COMM_WORLD, &recvRequests[5]);
			MPI_Recv_init(&grid[0][cols-1], 1, MPI_CHAR, northEast, 0, 		// receive NE cell
					MPI_COMM_WORLD, &recvRequests[6]);
			MPI_Recv_init(&grid[0][0], 1, MPI_CHAR, northWest, 0, 			// receive NW cell
					MPI_COMM_WORLD, &recvRequests[7]);
			// initiate all receive requests and wait for them to finish too before proceeding
			MPI_Startall(8, recvRequests);
			MPI_Waitall(8, recvRequests, recvStatuses);

			/*
			// 2nd Less efficient method
			MPI_Sendrecv(grid[1], cols, MPI_CHAR, north, 0,					// send top row
						grid[rows-1], cols, MPI_CHAR, south, 0, 			// receive south row
						MPI_COMM_WORLD, &sendRecvStatuses[0]);
			MPI_Sendrecv(grid[rows-2], cols, MPI_CHAR, south, 0, 			// send bottom row
						grid[0], cols, MPI_CHAR, north, 0, 					// receive north row
						MPI_COMM_WORLD, &sendRecvStatuses[1]);
			MPI_Sendrecv(sendLeftColumn, rows, MPI_CHAR, west, 0, 			// send left column
						recvRightColumn, rows, MPI_CHAR, east, 0, 			// receive right col
						MPI_COMM_WORLD, &sendRecvStatuses[2]);
			MPI_Sendrecv(sendRightColumn, rows, MPI_CHAR, east, 0, 			// send right column
						recvLeftColumn, rows, MPI_CHAR, west, 0, 			// receive left col
						MPI_COMM_WORLD, &sendRecvStatuses[3]);
			MPI_Sendrecv(&grid[1][1], 1, MPI_CHAR, northWest, 0, 			// send NW cell
						&grid[rows-1][cols-1], 1, MPI_CHAR, southEast, 0, 	// receive SE cell
						MPI_COMM_WORLD, &sendRecvStatuses[4]);
			MPI_Sendrecv(&grid[1][cols-2], 1, MPI_CHAR, northEast, 0, 		// send NE cell
						&grid[rows-1][0], 1, MPI_CHAR, southWest, 0, 		// receive SW cell
						MPI_COMM_WORLD, &sendRecvStatuses[5]);
			MPI_Sendrecv(&grid[rows-2][1], 1, MPI_CHAR, southWest, 0, 		// send SW cell
						&grid[0][cols-1], 1, MPI_CHAR, northEast, 0, 		// receive NE cell
						MPI_COMM_WORLD, &sendRecvStatuses[6]);
			MPI_Sendrecv(&grid[rows-2][cols-2], 1, MPI_CHAR, southEast, 0, 	// send SE cell
						&grid[0][0], 1, MPI_CHAR, northWest, 0, 			// receive NW cell
						MPI_COMM_WORLD, &sendRecvStatuses[7]);
			*/
		}
		else // if (np < 9)  
		// non-blocking communication for processes that may send / recv to themselves
		{
			// 1. Sending Rows & Columns
			MPI_Isend(grid[1], cols, MPI_CHAR, north, 0, 					// send top row
					MPI_COMM_WORLD, &sendRequests[0]);
			MPI_Isend(grid[rows-2], cols, MPI_CHAR, south, 0, 				// send bottom row
					MPI_COMM_WORLD, &sendRequests[1]);
			MPI_Isend(sendLeftColumn, rows, MPI_CHAR, west, 0, 				// send left column
					MPI_COMM_WORLD, &sendRequests[2]);
			MPI_Isend(sendRightColumn, rows, MPI_CHAR, east, 0, 			// send right column
					MPI_COMM_WORLD, &sendRequests[3]);
			// 2. Sending Corner Cells
			MPI_Isend(&grid[1][1], 1, MPI_CHAR, northWest, 0, 				// send NW cell
					MPI_COMM_WORLD, &sendRequests[4]);
			MPI_Isend(&grid[1][cols-2], 1, MPI_CHAR, northEast, 0, 			// send NE cell
					MPI_COMM_WORLD, &sendRequests[5]);
			MPI_Isend(&grid[rows-2][1], 1, MPI_CHAR, southWest, 0, 			// send SW cell
					MPI_COMM_WORLD, &sendRequests[6]);
			MPI_Isend(&grid[rows-2][cols-2], 1, MPI_CHAR, southEast, 0,		// send SE cell
					MPI_COMM_WORLD, &sendRequests[7]);
			// do NOT MPI_Waitall(8, sendRequests, sendStatuses);	// when sending to self
			// 3. Receiving Rows & Columns into the Ghost Rows & Columns
			MPI_Irecv(grid[rows-1], cols, MPI_CHAR, south, 0, 				// receive south row
					MPI_COMM_WORLD, &recvRequests[0]);
			MPI_Irecv(grid[0], cols, MPI_CHAR, north, 0, 					// receive north row
					MPI_COMM_WORLD, &recvRequests[1]);
			MPI_Irecv(recvRightColumn, rows, MPI_CHAR, east, 0, 			// receive right col
					MPI_COMM_WORLD, &recvRequests[2]);
			MPI_Irecv(recvLeftColumn, rows, MPI_CHAR, west, 0, 				// receive left col
					MPI_COMM_WORLD, &recvRequests[3]);
			// 4. Receiving Corner Cells into the 4 ghost corner cells
			MPI_Irecv(&grid[rows-1][cols-1], 1, MPI_CHAR, southEast, 0,		// receive SE cell
					MPI_COMM_WORLD, &recvRequests[4]);
			MPI_Irecv(&grid[rows-1][0], 1, MPI_CHAR, southWest, 0, 			// receive SW cell
					MPI_COMM_WORLD, &recvRequests[5]);
			MPI_Irecv(&grid[0][cols-1], 1, MPI_CHAR, northEast, 0, 			// receive NE cell
					MPI_COMM_WORLD, &recvRequests[6]);
			MPI_Irecv(&grid[0][0], 1, MPI_CHAR, northWest, 0, 				// receive NW cell
					MPI_COMM_WORLD, &recvRequests[7]);
			MPI_Waitall(8, recvRequests, recvStatuses);
		}

		#pragma omp parallel for shared(rows, cols, grid, recvRightColumn, recvLeftColumn) private(i) num_threads(4)
		for (i = 1; i < rows - 1; i++) 
		{
			grid[i][cols-1] = recvRightColumn[i];						// store right col
			grid[i][0] = recvLeftColumn[i];								// store left col
		}

		/****************************************/
		// B. Display Game Grid	//
#ifdef DISPLAY_RESULTS
		// MPI_Gatherv()-ing subarrays into the global array for display..
		
		// recvBlocks : # of units of the custom data type sendBlock, that each process will send
		#pragma omp parallel for shared(recvBlocks) private(i) num_threads(4)
		for (i = 0; i < np; i++)
			recvBlocks[i] = 1;

		// displacements: displacement relative to global buffer (universe) at which to place the
		//                             incoming data block from process i -- in block extents! --
		#pragma omp parallel num_threads(2)
		;
		i = 0;
		#pragma omp for private(prow, pcol)
		for (prow = 0; prow < prows; prow++)
		{
			for (pcol = 0; pcol < pcols; pcol++, i++)
				displacements[i] = pcol + prow * ((rows - 2) * pcols);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Gatherv(...) is a collective routine
		// Gathers the local arrays to the global array in specified locations
		// send type: sendBlock (the sender's array without the ghost cells)
		// recv type: recvBlock (the same type as above will be handled by the root process)
		errNo = 
		MPI_Gatherv(&grid[0][0], 1, sendBlock,								// senders params
					recvPtr, recvBlocks, displacements, recvBlock, master,	// receiver params
					MPI_COMM_WORLD);
		ASSERT(errNo, MPI_SUCCESS);
		
		// Display global array
		if (rank == master) 
			display2dGrid(universe, Rows, Cols, turn);

		sleep(timeStep);
#endif

		/****************************************/
		// C. Compute grid for the next generation	//
		newGrid = evolve(grid, rows, cols);

		// free memory allocated to previous evolution
		deallocContiguous2dArr(grid, gridData);
		grid = newGrid;	// now pointing to the newly created grid
		turn++;
	} // end main loop	//


	if (rank == master) 
	{
		printf("\nSimulation ended @gen #%d.\n", turn);
		printf("Elapsed time: %fs\n", MPI_Wtime() - startTime);
	}

	/********************************************************************************/
#if defined(_DEBUG) || defined(DEBUG)
	// write last grid state to a file
	writeGridToFile(universe, Rows, Cols);
#endif

	// VI. Deallocate resources & Terminate	//
#ifdef DISPLAY_RESULTS
	if (rank == master) 
	{
		MPI_Type_free(&recvBlock);
		MPI_Type_free(&pre_recvBlock);
		deallocContiguous2dArr(universe, universeData);
	}
	MPI_Type_free(&sendBlock);
#endif
	deallocContiguous2dArr(grid, gridData);

	MPI_Finalize();
	return 0;
}


///////////////////////////////////////////////////////////////////
//////////////////////////// Functions ////////////////////////////

// parse and process program's arguments
void parseCmd(int* argc, char*** argv) 
{
	int nargs = *argc;
	bool pause = false;

#ifdef DISPLAY_RESULTS
	if (rank == master)
	{
		printf("\n\n----------------------------------------------------------\n");
		printf    ("-------- Game of Life - MPI - OpenMP - Optimized ---------\n");
		printf    ("------------------ by Nikos Lazaridis --------------------\n");
		printf    ("----------------------------------------------------------\n");
		sleep(1000);
	}
#endif

	if (nargs < 5) 
	{
		help();
		pause = true;
	}

	if (nargs < 2) 
	{
		if (rank == 0) 
			printf("Setting rows = 24\n");
	}
	else 
		Rows = atoi((*argv)[1]);

	if (nargs < 3) 
	{
		if (rank == 0) 
			printf("Setting columns = 48\n");
	}
	else 
		Cols = atoi((*argv)[2]);

	if (nargs < 4) 
	{
		if (rank == 0) 
			printf("Setting generations = 1000\n");
	}
	else  
		generations = atoi((*argv)[3]);

	if (nargs < 5) 
	{
		if (rank == 0) 
			printf("Setting density = 0.25\n");
	}
	else 
		density = (double) atof((*argv)[4]);

	if (pause) 
		sleep(5000);

	// subdivide blocks to processes	//
	if (np >= 9) 
	{
		if ((np % 3) != 0)	// mandatory requirement
		{
			help();
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		// np = prows * pcols
		if (Rows <= Cols) 
		{
			prows = 3;
			pcols = np / prows;
		}
		else 
		{
			pcols = 3;
			prows = np / pcols;
		}
	}
	else // if (np < 9) 
	{
		if (Rows <= Cols) 
		{
			prows = 1;
			pcols = np;

			if ((Cols % np) != 0) 
			{
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		else 
		{
			prows = np;
			pcols = 1;

			if ((Rows % np) != 0) 
			{
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	rows = Rows / prows;
	cols = Cols / pcols;

	// Adding 2 rows and columns to account for the ghost cells
	rows += 2;
	cols += 2;
}


#ifdef DISPLAY_RESULTS
	void display2dGrid(char** grid, int nrows, int ncolumns, int turn)
	{
		updateScreen();
		#pragma omp parallel num_threads(4)
		;
		int y, x;
		#pragma omp for
		for (y = 0; y < nrows; y++) 
		{
			for (x = 0; x < ncolumns; x++) 
			{
				printf("%c", grid[y][x]);
			}
			printf("\n");
		}
		printf("Gen #%d\n", turn);
		fflush(stdout); // clear / flush specified buffer before the next operation
	}
#endif

#if defined(_DEBUG) || defined(DEBUG)
	void writeGridToFile(char** grid, int nrows, int ncolumns) 
	{
		FILE* fp = fopen("finalGrid.txt", "w");
		#pragma omp parallel num_threads(8)
		;
		#pragma omp for
		for (int y = 1; y < nrows - 1; y++) 
		{
			for (int x = 1; x < ncolumns - 1; x++) 
			{
				fprintf(fp, "%c", grid[y][x]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
#endif
// An MPI function assumes the passed array in memory is allocated contiguously!
// we need to adjust pointers of pointers data structure to point to contiguous memory (2D Array)
char** allocContiguous2dArr(char* gridData, int nrows, int ncolumns) 
{
	gridData = (char*)malloc(nrows * ncolumns * sizeof(char));
	char** grid = (char**)malloc(nrows * sizeof(char*));
	int i;
	for (i = 0; i < nrows; i++) 
	{
		grid[i] = &(gridData[i * ncolumns]);
	}

	return grid;
}

void deallocContiguous2dArr(char** grid, char* gridData) 
{
	free(gridData);
	free(grid);
}

char** randomlyPopulateGrid(char** grid, int nrows, int ncolumns) 
{
	srand(time(0) ^ (rank * 33 / 4));

	#pragma omp parallel num_threads(4) 
	;
	int y, x;
	#pragma omp for 
	for (y = 1; y < nrows - 1; y++) 
	{
		for (x = 1; x < ncolumns - 1; x++) 
		{
			if (((double)rand() / RAND_MAX) <= density) 
				grid[y][x] = '#';	// alive organism
			else 
				grid[y][x] = '.';	// dead cell
		}
	}

	return grid;
}

// counts alive cells in the Moore neighborhoud of the cell @ grid[y][x]
int countNeighbors(char** grid, int y, int x) 
{
	int aliveNeighbors = 0;
	char cell;
	/*
	. . . . .
	. X X X .
	. X X X .
	. X X X .
	. . . . .
	*/
	for (int i = -1; i <= 1; i++) 
	{
		for (int j = -1; j <= 1; j++) 
		{
			cell = grid[y + i][x + j];
			if (cell == '#') 
				aliveNeighbors++;
		}
	}
	// we don't want to include the current cell
	if (grid[y][x] == '#') 
		aliveNeighbors--;
#if defined(_DEBUG) || defined(DEBUG)
	if (aliveNeighbors > 8) 
		exit(8);
#endif

	return aliveNeighbors;
}

// process the grid for the next generation and create the new one according to 
// 	the 4 rules of the game
char** evolve(char** grid, int nrows, int ncolumns) 
{
	char* newGridData = NULL;
	char** newGrid = allocContiguous2dArr(newGridData, nrows, ncolumns);// local array for next gen
#if defined(_DEBUG) || defined(DEBUG)
	ASSERT(newGrid != NULL, 1);
#endif

	#pragma omp parallel num_threads(4) 
	;// the following variables will be local / private to the thread
	int nliving;
	char cell; 		// current grid cell
	char* newCell;	// points to corresponding cell of the newGrid

	#pragma omp for 
	// skipping the first and last rows (y) and columns (x) of the grid
	for (int y = 1; y < nrows - 1; y++) 
	{
		for (int x = 1; x < ncolumns - 1; x++) 
		{
			// determine alive neighbours of cell
			cell = grid[y][x];
			nliving = countNeighbors(grid, y, x);

		 	// determine the state of the cell in the new grid
			newCell = &newGrid[y][x];
			if (nliving == 3 || (nliving == 2 && cell == '#')) 
				*newCell = '#';
			else 
				*newCell = '.';
		}
	} // O(rows * columns)

	return newGrid;
}

// The array is periodic. There are always 8 neighbors to each block
void determineNeighbourRanks() 
{
	if (np >= 9) 
	{
		// a. Processes with blocks located @ 1 of the 4 corners of the block array
		if (rank == 0) 
		{ // top left block
			north = (prows - 1) * pcols; south = pcols;
			west = pcols - 1; east = 1;
			southEast = south + 1; southWest = 2 * pcols - 1;
			northEast = north + 1; northWest = np - 1;
		}
		else if (rank == np - 1) 
		{ // bottom right block
			north = rank - pcols; south = pcols - 1;
			west = rank - 1; east = np - pcols;
			northWest = north - 1; northEast = np - 2 * pcols;
			southWest = south - 1; southEast = 0;
		}
		else if (rank == pcols - 1) 
		{ // top right block
			north = np - 1; south = 2 * pcols - 1;
			west = rank - 1; east = 0;
			southWest = south - 1; southEast = pcols;
			northWest = north - 1; northEast = np - pcols;
		}
		else if (rank == np - pcols) 
		{ // bottom left corner
			north = rank - pcols; south = 0;
			west = np - 1; east = rank + 1;
			northWest = rank - 1; northEast = north + 1;
			southWest = pcols - 1; southEast = 1;
		}
		// b. Processes on the edges
		else if (rank < pcols - 1) 
		{ // Processes @ Top prow
			north = rank + (prows - 1) * pcols; south = rank + pcols;
			west = rank - 1; east = rank + 1;
			northWest = north - 1; northEast = north + 1;
			southWest = south - 1; southEast = south + 1;
		}
		else if (rank > np - pcols) 
		{ // Process @ Bottom prow
			north = rank - pcols; south = rank - (prows - 1) * pcols;
			west = rank - 1; east = rank + 1;
			northWest = north - 1; northEast = north + 1;
			southWest = south - 1; southEast = south + 1;
		}
		else if ((rank + 1) % pcols == 0) 
		{ // Processes @ right pcolumn
			north = rank - pcols; south = rank + pcols;
			west = rank - 1; east = north + 1;
			northWest = north - 1; northEast = east - pcols;
			southWest = south - 1; southEast = rank + 1;
		}
		else if ((rank % pcols) == 0) 
		{ // Processes @ left pcolumn
			north = rank - pcols; south = rank + pcols;
			west = rank + pcols - 1; east = rank + 1;
			northWest = rank - 1; northEast = north + 1;
			southWest = west + pcols; southEast = south + 1;
		}
		// c. Processes envelopped inside the edge blocks (not touching an outer edge)
		else 
		{
			north = rank - pcols; south = rank + pcols;
			west = rank - 1; east = rank + 1;
			northWest = north - 1; northEast = north + 1;
			southWest = south - 1; southEast = south + 1;
		}
	}
	else // if (np < 9) 
	{
		if (Rows <= Cols) 
		{
			north = south = rank;
			northWest = southWest = west = rank ? rank - 1 : np - 1;
			northEast = southEast = east = (rank + 1) % np;
		}
		else // if (Rows < Cols) 
		{
			northWest = northEast = north = rank ? rank - 1 : np - 1;
			southWest = southEast = south = (rank + 1) % np;
			west = east = rank;
		}
	}
}