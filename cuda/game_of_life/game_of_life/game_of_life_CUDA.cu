//***************************************************************************************//
// Game of Life - CUDA GPU Implementation
// Nikos Lazaridis, 
// University of Athens (UOA), Department of Informatics (DIT)
//***************************************************************************************//

//////////////////////////////////////////////////////////////////
//////////////////////////// Includes ////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>				// stops underlining of __global__ mems etc.
#include <device_launch_parameters.h>	// stops underlining of threadId etc.
#include "bypassIntellisense.h"

#include <iostream>
#include <fstream>
#if defined _WIN32 || defined _WIN64
#	include <Windows.h>// for Sleep()
#	include <ctime>
#	define sleep Sleep;
#elif defined __unix__ || defined __linux__
#	include <sys/time.h>
#	include <unistd.h> // for usleep()
#	define sleep( ms ) usleep( ms * 1000 );
#endif

/////////////////////////////////////////////////////////////////
//////////////////////////// Globals ////////////////////////////
// unified for global memory
#define BLOCK_SIZE 32
// separate handling of blocks for shared memory
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

#if defined( _DEBUG ) || defined( DEBUG )
#	define ASSERT( ret, num ) \
		do \
		{ \
			if (ret != num) \
			{ \
				fprintf( stderr,\
						"\nerrNo [%d] at line [%d] in function [%s]. Date: [%s] ",\
						ret,\
						__LINE__,\
						__func__,\
						__DATE__ );\
				exit( ret ); \
			} \
		} while ( 0 )
#else // NDEBUG
#	define ASSERT(ret, num)
#endif


#ifdef DISPLAY_RESULTS
extern "C"
void displayGrid( char* grid,
	int dimCells )
{
	for ( int y = 1; y < dimCells + 1; y++ )
	{
		for ( int x = 1; x < dimCells + 1; x++ )
		{
			std::cout << grid[y * ( dimCells + 2 ) + x] << " ";
		}
		std::cout << std::endl;
	}
	fflush( stdout ); // flush output buffer
}
#endif

extern "C"
void writeGridToFile( char* grid,
	int dimCells )
{
	std::ofstream fp( "final_cuda_Grid.txt" );
	for ( int y = 1; y < dimCells + 1; y++ )
	{
		for ( int x = 1; x < dimCells + 1; x++ )
		{
			fp << grid[y * ( dimCells + 2 ) + x];
		}
		fp << std::endl;
	}
	fp.close();
}

__global__ void ghostRows( int dimCells,
	char* d_grid )
{
	// tid = [1, dimCells]
	int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;
	// first real row (@ index 1) goes to bottom ghost row
	d_grid[( dimCells + 2 ) * ( dimCells + 1 ) + tid] = d_grid[( dimCells + 2 ) + tid];
	// last real row goes to top ghost row
	d_grid[tid] = d_grid[( dimCells + 2 ) * dimCells + tid];
}

// also handles ghost columns
__global__ void copyGhostCols(int dimCells, char* d_grid)
{
	// tid = [0, dimCells+1]
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// first real column goes to right most ghost column
	d_grid[tid * ( dimCells + 2 ) + dimCells + 1] = d_grid[tid * ( dimCells + 2 ) + 1];
	// last real column goes to left most ghost column 
	d_grid[tid * ( dimCells + 2 )] = d_grid[tid * ( dimCells + 2 ) + dimCells];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef CUDA_SHARED
// evolves the grid
__global__ void evolveShared( int dimCells,
	char* d_grid,
	char* d_newGrid )
{
	// tid = [1, dimCells]
	// each thread block will contain 2 ghost rows and 2 ghost columns
	// meaning that (BLOCK_SIZE_Y - 2) * (BLOCK_SIZE_X - 2) elements will be processed 
	// 	in each kernel run
	int tid_y = blockIdx.y * ( blockDim.y - 2 ) + threadIdx.y;
	int tid_x = blockIdx.x * ( blockDim.x - 2 ) + threadIdx.x;
	int tid = tid_y * ( dimCells + 2 ) + tid_x;

	// "coordinates" of thread within the block
	int i = threadIdx.y;	// ROW INDEX
	int j = threadIdx.x;	// COLUMN INDEX

	// 1. shared video memory segment for each block
	__shared__ char sharedGrid[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	// 2. copy cells into shared memory
	sharedGrid[i][j] = d_grid[tid];
	// 3. synchronize before moving on
	__syncthreads();

	if ( i != 0 && i != blockDim.y - 1 && j != 0 && j != blockDim.x - 1 )
	{
		char currentCell = sharedGrid[i][j];
		int nliving = 0;
		// counting alive neighbours:
		for ( int y = i - 1; y <= i + 1; y++ )
		{
			for ( int x = j - 1; x <= j + 1; x++ )
			{
				if ( sharedGrid[y][x] == '#' )
					nliving++;
			}
		}

		// we don't want to include the current cell in the counting
		if ( currentCell == '#' )
			nliving--;

		// determine the state of the cell in the new grid based on the rules of the game
		if ( nliving == 3 || ( nliving == 2 && currentCell == '#' ) )
			d_newGrid[tid] = '#';	// cell survives
		else
			d_newGrid[tid] = '.';	// cell dies
	}
}

// time to play the game (by Motorhead :D )
extern "C"
void playGameShared( char* d_grid,
	char* d_newGrid,
	int dimCells,
	int generations )
{
	char* d_tmpGrid;// temporary grid pointer used to switch between d_grid and d_newGrid

	// block and grid sizes for dispatching threads to the copyGhostCells() kernel
	dim3 cpyBlockDims( BLOCK_SIZE_X,
		1,
		1 );
	dim3 cpyGhostRowsDims( static_cast<int>( ceil( dimCells
			/ static_cast<float>( cpyBlockDims.x ) ) ),
		1,
		1 );
	dim3 cpyGhostColumnsDims( static_cast<int>( ceil( ( dimCells + 2 )
			/ static_cast<float>( cpyBlockDims.x ) ) ),
		1,
		1 );
	// block and grid sizes for dispatching threads to the evolve() kernel - 1024 thrs max
	dim3 blockDims( BLOCK_SIZE_X,
		BLOCK_SIZE_Y,
		1 );
	dim3 gridDims( static_cast<int>( ceil( dimCells
			/ static_cast<float>( blockDims.x - 2 ) ) ),
		static_cast<int>( ceil( dimCells
			/ static_cast<float>( blockDims.y - 2 ) ) ),
		1 ) ;


	// count GPU execution time with CUDA utilities
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	float msElapsedInGpu;
	// create timing event objects
	cudaEventCreate( &startTime );
	cudaEventCreate( &stopTime );
	cudaEventRecord( startTime,
		0 );

	// game loop
	for ( int turn = 0; turn < generations; turn++ )
	{
		// I. Display grid
#ifdef DISPLAY_RESULTS
		if ( dimCells < 1025 )
			displayGrid( d_grid,
				dimCells );
#endif

		// II. Copy ghost cells
		ghostRows<<<cpyGhostRowsDims, cpyBlockDims>>>( dimCells, d_grid );
		copyGhostCols<<<cpyGhostColumnsDims, cpyBlockDims>>>( dimCells, d_grid );
		evolveShared<<<gridDims, blockDims>>>( dimCells, d_grid, d_newGrid );
		//cudaDeviceSynchronize();

		// III. Swap grids
		d_tmpGrid = d_grid;
		d_grid = d_newGrid;
		d_newGrid = d_tmpGrid;
	}

	cudaEventRecord( stopTime, 0 );		// record stopTime
	cudaEventSynchronize( stopTime );	// wait for stopTime event to be completed
	cudaEventElapsedTime( &msElapsedInGpu,
		startTime,
		stopTime );
	std::cout << "Time elapsed in GPU computations with shared memory utilization: "
		<< msElapsedInGpu
		<< "ms.\n\n";
}

#else

// this kernel will produce the next grid
__global__ void evolveGlobal( int dimCells,
	char* d_grid,
	char* d_newGrid )
{
	// tid = [1, dimCells]
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int tid = tid_y * ( dimCells + 2 ) + tid_x;

	char currentCell = d_grid[tid];
	int nliving = 0;
	// get number of alive neighbouring cells
	for ( int y = -1; y <= 1; y++ )
	{
		for ( int x = -1; x <= 1; x++ )
		{
			if ( d_grid[tid + ( y * ( dimCells + 2 ) ) + x] == '#' )
				nliving++;
		}
	}

	// we don't want to include the current cell in the counting
	if ( currentCell == '#' )
		nliving--;

	// determine the state of the cell in the new grid based on the rules of the game
	if ( nliving == 3 || ( nliving == 2 && currentCell == '#' ) )
		d_newGrid[tid] = '#';
	else
		d_newGrid[tid] = '.';
}

// start GOL simulation using global vmemory segment
// arguments are device pointers - unable to be dereferenced by the host machine
extern "C"
void playGameGlobal( char* d_grid,
	char* d_newGrid,
	int dimCells,
	int generations )
{
	char* d_tmpGrid;// temporary grid pointer used to switch between d_grid and d_newGrid

	// block and grid sizes for dispatching threads to the copyGhostCells() kernel
	dim3 cpyBlockDims( BLOCK_SIZE,
		1,
		1 );
	dim3 cpyGhostRowsDims( static_cast<int>( ceil( dimCells / 
		static_cast<float>( cpyBlockDims.x ) ) ),
		1,
		1 );
	dim3 cpyGhostColumnsDims( static_cast<int>( ceil( ( dimCells + 2 ) / 
		static_cast<float>( cpyBlockDims.x ) ) ),
		1,
		1 );
	// block and grid sizes for dispatching threads to the evolve() kernel
	dim3 blockDims( BLOCK_SIZE,
		BLOCK_SIZE,
		1 );	// block dimensions (in every direction)
	dim3 gridDims(
		static_cast<int>( ceil( dimCells / static_cast<float>( blockDims.x ) ) ),
		static_cast<int>( ceil( dimCells / static_cast<float>( blockDims.y ) ) ),
		1 );


	// count GPU execution time with CUDA utilities
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	float msElapsedInGpu;
	// create timing event objects
	cudaEventCreate( &startTime );
	cudaEventCreate( &stopTime );
	cudaEventRecord( startTime, nullptr );

	// game loop
	for ( int turn = 0; turn < generations; turn++ )
	{
		// I. Display grid
#	ifdef DISPLAY_RESULTS
		if ( dimCells < 1025 )
			displayGrid( d_grid,
				dimCells );
#	endif
		// II. Copy ghost cells
		ghostRows<<<cpyGhostRowsDims, cpyBlockDims>>>( dimCells, d_grid );
		copyGhostCols<<<cpyGhostColumnsDims, cpyBlockDims>>>( dimCells, d_grid );
		evolveGlobal<<<gridDims, blockDims>>>( dimCells, d_grid, d_newGrid );
		//cudaDeviceSynchronize();

		// III. Swap grids
		d_tmpGrid = d_grid;
		d_grid = d_newGrid;
		d_newGrid = d_tmpGrid;
	}

	cudaEventRecord( stopTime, nullptr );	// record stopTime
	cudaEventSynchronize( stopTime );		// wait for stopTime event to be completed
	cudaEventElapsedTime( &msElapsedInGpu,	// compute time elapsed
		startTime,
		stopTime );
	std::cout << "Time elapsed in GPU computations with global memory utilization: "
		<< msElapsedInGpu
		<< "ms. \n\n";
}
#endif

extern "C" void help()
{
	std::cout << "Call:\n./[programName] [#elementsInEachDimension] [ngenerations] "
		"[density(0-1.0)]\n\n";
}


int main( int argc,
	char* argv[] )
{
	char* grid;				// grid on host
	char* initialGrid;		// backup array depicting initial grid status (for testing)
	char* d_grid;			// grid on device
	char* d_newGrid;		// new grid used on device only
	int dimCells = 64;		// cell count in each dimension, excluding ghost cells
	int ngenerations = 2000;// max number of generations
	double density = 0.25;

	//////////////////////////////////////////////////////////////////////////////////////
	// process command line arguments
	bool pause = false;
	if ( argc < 4 )
	{
		help();
		pause = true;
	}

	// process command line arguments
	if ( argc < 2 )
		std::cout << "Dimension must be greater than zero. Setting to 64\n";
	else
		dimCells = atoi( argv[1] );

	if ( argc < 3 )
		std::cout << "Setting generations to 2000\n";
	else
		ngenerations = atoi( argv[2] );

	if ( argc < 4 )
		std::cout << "Setting density to 0.25\n";
	else
		density = (double)atof( argv[3] );

	if ( pause )
		sleep( 2000 );
	////////////////////////////////////////////////////////////////////////////////////////////////


	// allocate space for the grids both in RAM and VRAM
	// (adjusted to include the ghost cells)
	unsigned count = ( dimCells + 2 ) * ( dimCells + 2 );
	grid = new char[count];
	initialGrid = new char[count];
	cudaMalloc( &d_grid, count );
	cudaMalloc( &d_newGrid, count );

	// Assign initial population randomly
	srand( static_cast<unsigned>( time( nullptr ) * 33 / 17) );
	for ( int i = 1; i < dimCells + 1; i++ )
	{
		for ( int j = 1; j < dimCells + 1; j++ )
		{
			if ( (double)rand() / RAND_MAX <= density )
				initialGrid[i * ( dimCells + 2 ) + j] =
				grid[i * ( dimCells + 2 ) + j] = '#';	// alive organism
			else
				initialGrid[i * ( dimCells + 2 ) + j] =
				grid[i * ( dimCells + 2 ) + j] = '.';	// dead cell
		}
	}
	// copy grid to VRAM (d_grid)
	cudaMemcpy( d_grid,
		grid,
		count,
		cudaMemcpyHostToDevice );

	// run
#ifdef CUDA_SHARED
	// function attribute request for preferred cache configuration
	cudaFuncSetCacheConfig( evolveShared, cudaFuncCachePreferShared );
	playGameShared( d_grid, d_newGrid, dimCells, ngenerations );
#else
	playGameGlobal( d_grid, d_newGrid, dimCells, ngenerations );
#endif

	// copy grid back to CPU RAM
	cudaMemcpy( grid,
		d_grid,
		count,
		cudaMemcpyDeviceToHost );

	int errNo = cudaGetLastError();
	ASSERT( errNo, cudaSuccess );

	writeGridToFile( grid,
		dimCells );

	// count alive cells
	int nalive = 0;
	for ( int i = 1; i < dimCells + 1; i++ )
	{
		for ( int j = 1; j < dimCells + 1; j++ )
		{
			if ( grid[i * ( dimCells + 2 ) + j] == '#' )
				nalive++;
		}
	}
	std::cout << "Remaining cells alive: "
		<< nalive
		<< " in generation "
		<< ngenerations << std::endl;

	// Release memory
	delete[] grid;
	cudaFree( d_grid );
	cudaFree( d_newGrid );

	return 0;
}