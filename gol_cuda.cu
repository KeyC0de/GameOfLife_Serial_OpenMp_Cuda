//***************************************************************************************//
// Game of Life - CUDA Implementation
// Nikos Lazaridis, M1485
// University of Athens (UoA), Department of Informatics (DIT),
// Parallel Computing Systems semester project
//***************************************************************************************//

//////////////////////////////////////////////////////////////////
//////////////////////////// Includes ////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>				// stops underlining of __global__ mems etc.
#include <device_launch_parameters.h>	// stops underlining of threadId etc.
#include "bypassIntellisense.h"

#include <iostream>
#include <fstream>
#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>// for Sleep()
#include <time.h>
#define sleep Sleep
#elif defined (__unix__) || defined (__linux__)
#include <sys/time.h>
#include <unistd.h> // for usleep()
#define sleep(ms) usleep(ms * 1000)
#endif

/////////////////////////////////////////////////////////////////
//////////////////////////// Globals ////////////////////////////
// thread block sizes
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

#if defined(_DEBUG) || defined(DEBUG)
	#define ASSERT(ret, num) \
			do \
			{ \
				if (ret != num) \
				{ \
					fprintf(stderr, "\nerrNo [%d] at line [%d] in function [%s]. Date: [%s] ", \
							ret, __LINE__, __func__, __DATE__); \
					exit(ret); \
				} \
			} while (0)
#else // NDEBUG
	#define ASSERT(ret, num)
#endif

//////////////////////////////////////////////////////////////////
///////////////////// Function Declarations //////////////////////
#ifdef DISPLAY_RESULTS
	extern "C" void displayGrid(char* grid, int rows, int cols);
#endif
#if defined(_DEBUG) || defined(DEBUG)
	extern "C" void writeGridToFile(char* grid, int rows, int cols);
#endif
#ifdef CUDA_SHARED
	__global__ void evolve_shared(char* d_grid, char* d_newGrid, int rows, int cols);
	extern "C" void playGame_shared(char* d_grid, char* d_newGrid, 
									int generations, int rows, int cols);
#else
	__global__ void evolve_global(char* d_grid, char* d_newGrid, int cols);
	extern "C" void playGame_global(char* d_grid, char* d_newGrid, 
									int generations, int rows, int cols);
#endif
__global__ void copyGhostRows(char* d_grid, int rows, int cols);
__global__ void copyGhostCols(char* d_grid, int rows, int cols);
extern "C" void help();


//////////////////////////////////////////////////////////////////
/////////////////////////// Functions ////////////////////////////
#ifdef DISPLAY_RESULTS
	extern "C" void displayGrid(char* grid, int rows, int cols)
	{
		for (int y = 1; y < rows + 1; y++)
		{
			for (int x = 1; x < cols + 1; x++)
			{
				std::cout << grid[y * (cols + 2) + x] << " ";
			}
			std::cout << std::endl;
		}
		fflush(stdout); // flush output buffer
	}
#endif

#if defined(_DEBUG) || defined(DEBUG)
	extern "C" void writeGridToFile(char* grid, int rows, int cols)
	{
		std::ofstream fp("final_cuda_grid.txt");
		for (int y = 1; y < rows + 1; y++)
		{
			for (int x = 1; x < cols + 1; x++)
			{
				fp << grid[y * (cols + 2) + x];
			}
			fp << std::endl;
		}
		fp.close();
	}
#endif

__global__ void copyGhostRows(char* d_grid, int rows, int cols)
{
	// tid = [1, cols]
	int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;
	
	// first real row (@ index 1) goes to bottom ghost row
	d_grid[(rows + 1) * (cols + 2) + tid] = d_grid[(cols + 2) + tid];
	// last real row goes to top ghost row
	d_grid[tid] = d_grid[rows * (cols + 2) + tid];
}

__global__ void copyGhostCols(char* d_grid, int rows, int cols)
{
	// tid = [0, rows+1]
	int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// first real column goes to right most ghost column
	d_grid[tid * (cols + 2) + cols + 1] = d_grid[tid * (cols + 2) + 1];
	// last real column goes to left most ghost column 
	d_grid[tid * (cols + 2)] = d_grid[tid * (cols + 2) + cols];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef CUDA_SHARED
// evolves the grid
__global__ void evolve_shared(char* d_grid, char* d_newGrid, int rows, int cols)
{
	// each thread block will contain 2 ghost rows and 2 ghost columns
	// meaning that (BLOCK_SIZE_Y - 2) rows * (BLOCK_SIZE_X - 2) cols will be processed in each run
	// [INDEX = row_index * row_size + col_index]
	int tid_x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
	int tid_y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;
	int tid = tid_y * (cols + 2) + tid_x;

	// "coordinates" of thread within the block
	int ti = threadIdx.x;	// Thread column index in block
	int tj = threadIdx.y;	// Thread row index in block
	int nliving;

	// 1. shared video memory segment for each block
	__shared__ char sharedGrid[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	// 2. copy cells into shared memory
	if (tid_x < cols + 2 && tid_y < rows + 2)
		sharedGrid[tj][ti] = d_grid[tid];
	// 3. synchronize before moving on
	__syncthreads();


	if (tid_x < cols + 1 && tid_y < rows + 1)
	{
		if (ti != 0 && ti != blockDim.x - 1 && tj != 0 && tj != blockDim.y - 1)
		{
			char currentCell = sharedGrid[tj][ti];
			nliving = 0;
			// counting alive neighbours:
			for (int y = tj - 1; y <= tj + 1; y++)
			{
				for (int x = ti - 1; x <= ti + 1; x++)
				{
					if (sharedGrid[y][x] == '#')
						nliving++;
				}
			}

			// we don't want to include the current cell in the counting
			if (currentCell == '#')
				nliving--;

			// determine the state of the cell in the new grid based on the rules of the game
			if (nliving == 3 || (nliving == 2 && currentCell == '#'))
				d_newGrid[tid] = '#';	// cell survives
			else
				d_newGrid[tid] = '.';	// cell dies
		}
	}
}

// time to play the game (by Motorhead :D )
extern "C" void playGame_shared(char* d_grid, char* d_newGrid, int generations, int rows, int cols)
{
	char* d_tmpGrid;	// temporary grid pointer used to switch between d_grid and d_newGrid

	// block and grid sizes for dispatching threads to the copyGhostRows() kernel
	dim3 copyRowsBlock(BLOCK_SIZE_X, 1, 1);
	dim3 copyRowsGrid(
		static_cast<int>(ceil((cols + 2) / static_cast<float>(copyRowsBlock.x))),
		1,
		1
	);
	
	// block and grid sizes for dispatching threads to the copyGhostCols() kernel
	dim3 copyColsBlock(BLOCK_SIZE_Y, 1, 1);
	dim3 copyColsGrid(
		static_cast<int>(ceil((rows + 2) / static_cast<float>(copyColsBlock.x))),
		1,
		1
	);

	// block and grid sizes for dispatching threads to the evolve_shared() kernel
	dim3 gameBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);	// block dimensions (in every direction)
	dim3 gameGrid(
		static_cast<int>(ceil(cols / static_cast<float>(gameBlock.x - 2))),
		static_cast<int>(ceil(rows / static_cast<float>(gameBlock.y - 2))),
		1
	);


	// count GPU execution time with CUDA utilities
	cudaEvent_t startTime, stopTime;
	float elapsedTime;
	// create timing event objects
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime, 0);	// start counting, by recording the startTime

	// game loop
	for (int turn = 0; turn < generations; turn++)
	{
		// I. Display grid
#ifdef DISPLAY_RESULTS
		displayGrid(d_grid, rows, cols);
#endif
		// II. Copy ghost cells
		copyGhostRows <<< copyRowsGrid, copyRowsBlock >>>(d_grid, rows, cols);
		copyGhostCols <<< copyColsGrid, copyColsBlock >>>(d_grid, rows, cols);
		evolve_shared <<< gameGrid, gameBlock >>>(d_grid, d_newGrid, rows, cols);
		//cudaDeviceSynchronize();

		// III. Swap grids
		d_tmpGrid = d_grid;
		d_grid = d_newGrid;
		d_newGrid = d_tmpGrid;
	}

	cudaEventRecord(stopTime, 0);	// record stopTime
	cudaEventSynchronize(stopTime);	// wait for stopTime event to be completed
									// compute time elapsed based on the recordings
	cudaEventElapsedTime(&elapsedTime, startTime, stopTime);
	std::cout << "Time elapsed in GPU computations with GPU shared memory utilization: "
		<< elapsedTime << "ms.\n\n";
}

#else

// this kernel will produce the next grid
__global__ void evolve_global(char* d_grid, char* d_newGrid, int cols)
{
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int tid = tid_y * (cols + 2) + tid_x;

	char currentCell = d_grid[tid];
	int nliving = 0;
	// get number of alive neighbouring cells
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			if (d_grid[tid + (y * (cols + 2)) + x] == '#')
				nliving++;
		}
	}

	// we don't want to include the current cell in the counting
	if (currentCell == '#')
		nliving--;

	// determine the state of the cell in the new grid based on the rules of the game
	if (nliving == 3 || (nliving == 2 && currentCell == '#'))
		d_newGrid[tid] = '#';
	else
		d_newGrid[tid] = '.';
}

// start GOL simulation using global vmemory segment
// arguments are device pointers - unable to be dereferenced by the host machine
extern "C" void playGame_global(char* d_grid, char* d_newGrid, int generations, int rows, int cols)
{
	char* d_tmpGrid;	// temporary grid pointer used to switch between d_grid and d_newGrid

	// block and grid sizes for dispatching threads to the copyGhostRows() kernel
	dim3 copyRowsBlock(BLOCK_SIZE_X, 1, 1);
	dim3 copyRowsGrid(
		static_cast<int>(ceil((cols + 2) / static_cast<float>(copyRowsBlock.x))),
		1,
		1
	);
	
	// block and grid sizes for dispatching threads to the copyGhostCols() kernel
	dim3 copyColsBlock(BLOCK_SIZE_Y, 1, 1);
	dim3 copyColsGrid(
		static_cast<int>(ceil((rows + 2) / static_cast<float>(copyColsBlock.x))),
		1,
		1
	);

	// block and grid sizes for dispatching threads to the evolve() kernel
	dim3 gameBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);	// block dimensions (in every direction)
	dim3 gameGrid(
		static_cast<int>(ceil(cols / static_cast<float>(gameBlock.x))),
		static_cast<int>(ceil(rows / static_cast<float>(gameBlock.y))),
		1
	);


	// count GPU execution time with CUDA utilities
	cudaEvent_t startTime, stopTime;
	float elapsedTime;
	// create timing event objects
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime, 0);	// start counting, by recording the startTime

	// game loop
	for (int turn = 0; turn < generations; turn++)
	{
		// I. Display grid
#ifdef DISPLAY_RESULTS
		displayGrid(d_grid, rows, cols);
#endif
		// II. Copy ghost cells
		copyGhostRows <<< copyRowsGrid, copyRowsBlock >>>(d_grid, rows, cols);
		copyGhostCols <<< copyColsGrid, copyColsBlock >>>(d_grid, rows, cols);
		evolve_global <<< gameGrid, gameBlock >>>(d_grid, d_newGrid, cols);
		//cudaDeviceSynchronize();

		// III. Swap grids
		d_tmpGrid = d_grid;
		d_grid = d_newGrid;
		d_newGrid = d_tmpGrid;
	}

	cudaEventRecord(stopTime, 0);	// record stopTime
	cudaEventSynchronize(stopTime);	// wait for stopTime event to be completed
	// compute time elapsed based on the recordings
	cudaEventElapsedTime(&elapsedTime, startTime, stopTime);
	std::cout << "Time elapsed in GPU computations with GPU global memory utilization: "
		<< elapsedTime << "ms. \n\n";
}
#endif

extern "C" void help()
{
	std::cout << "Call:\n./[programName] [#elementsInEachDimension] [ngenerations] " \
		"[density(0-1.0)]\n\n";
}


//////////////////////////////////////////////////////////////////
////////////////////////////   main   ////////////////////////////
int main(int argc, char* argv[])
{
	char* grid;				// grid on host
	char* initialGrid;		// backup array depicting initial grid status (for testing)
	char* d_grid;			// grid on device
	char* d_newGrid;		// new grid used on device only
	int rows = 128;
	int cols = 1024;
	int ngenerations = 1000;
	double density = 0.25;

	////////////////////////////////////////////////////////////////////////////////////////////////
	// process command line arguments
	bool pause = false;
	if (argc < 5)
	{
		help();
		pause = true;
	}

	if (argc < 2)
		std::cout << "Rows must be greater than zero. Setting to 128\n";
	else
		rows = atoi(argv[1]);

	if (argc < 3)
		std::cout << "Columns must be greater than zero. Setting to 1024\n";
	else
		cols = atoi(argv[2]);
	
	if (argc < 4)
		std::cout << "Setting generations to 1000\n";
	else
		ngenerations = atoi(argv[3]);

	if (argc < 5)
		std::cout << "Setting density to 0.25\n";
	else
		density = (double)atof(argv[4]);

	if (pause)
		sleep(2000);
	////////////////////////////////////////////////////////////////////////////////////////////////


	// allocate space for the grids both in RAM and VRAM
	unsigned count = (rows + 2) * (cols + 2);	//adjusted to include the ghost cells
	grid = new char[count];
	initialGrid = new char[count];
	cudaMalloc(&d_grid, count);
	cudaMalloc(&d_newGrid, count);

	// Assign initial population randomly
	srand(static_cast<unsigned>(time(0) * 33 / 17));
	for (int j = 1; j < rows + 1; j++)
	{
		for (int i = 1; i < cols + 1; i++)
		{
			if ((double)rand() / RAND_MAX <= density)
				initialGrid[j * (cols + 2) + i] = 
					grid[j * (cols + 2) + i] = '#';	// alive organism
			else
				initialGrid[j * (cols + 2) + i] = 
					grid[j * (cols + 2) + i] = '.';	// dead cell
		}
	}
	// copy grid to VRAM (d_grid)
	cudaMemcpy(d_grid, grid, count, cudaMemcpyHostToDevice);

	// run
#ifdef CUDA_SHARED
	// function attribute request for preferred cache configuration
	cudaFuncSetCacheConfig(evolve_shared, cudaFuncCachePreferShared);
	playGame_shared(d_grid, d_newGrid, ngenerations, rows, cols);
#else
	playGame_global(d_grid, d_newGrid, ngenerations, rows, cols);
#endif

	// copy grid back to CPU RAM
	cudaMemcpy(grid, d_grid, count, cudaMemcpyDeviceToHost);

	int errNo = cudaGetLastError();
	ASSERT(errNo, cudaSuccess);

#if defined(_DEBUG) || defined(DEBUG)
	// write last grid state to a file
	writeGridToFile(grid, rows, cols);
	
	// count alive cells
	int nalive = 0;
	for (int j = 1; j < rows + 1; j++)
	{
		for (int i = 1; i < cols + 1; i++)
		{
			if (grid[j * (cols + 2) + i] == '#')
				nalive++;
		}
	}
	std::cout << "Remaining cells alive: " << nalive << " in generation " 
				<< ngenerations << std::endl;
#endif

	// Release memory
	delete[] grid;
	cudaFree(d_grid);
	cudaFree(d_newGrid);

	return 0;
}



