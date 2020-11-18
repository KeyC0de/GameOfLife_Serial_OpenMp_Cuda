##########################################################################################
## Game of Life - Makefile																 #	##																						 #
## Nikos Lazaridis, M1485																 #	##																						 #
## University of Athens (UoA), Department of Informatics (DIT),							 #	##																						 #
## Parallel Computing Systems semester project											 #	##																						 #
##########################################################################################

# set bash shell as the make shell (sh is the default)
SHELL:=/bin/bash
CC = gcc
CFLAGS = -std=c11
CUDAFLAGS = -std=c++11 -arch=sm_30
OBJECTS = gol_serial.o gol_serial_alt.o gol_serial_openmp.o gol_serial_alt_openmp.o gol_mpi.o gol_mpi_openmp.o gol_cuda.o gol_cuda_shared.o
EXECUTABLES = gol_serial.exe gol_serial_alt.exe gol_serial_openmp.exe gol_serial_alt_openmp.exe gol_mpi.exe gol_mpi_openmp.exe gol_cuda.exe gol_cuda_shared.exe gol_serial gol_serial_alt gol_serial_openmp gol_serial_alt_openmp gol_mpi gol_mpi_openmp gol_cuda gol_cuda_shared


#######################
# build configuration #

# find out os
ifeq ($(OS),Windows_NT)
	CFLAGS += -D _WIN32
	CUDAFLAGS += -D _WIN32

	# query processor architecture
	ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
		CFLAGS += -D AMD64 -D _WIN64
		CUDAFLAGS += -D AMD64 -D _WIN64
	else
		ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
			CFLAGS += -D AMD64 -D _WIN64
			CUDAFLAGS += -D AMD64 -D _WIN64
		endif
		ifeq ($(PROCESSOR_ARCHITECTURE),x86)
			CFLAGS += -D IA32 -D _WIN32
			CUDAFLAGS += -D IA32 -D _WIN32
		endif
	endif
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CFLAGS += -D __unix__ -D __linux__ -D _BSD_SOURCE
		CUDAFLAGS += -D __unix__ -D __linux__ -D _BSD_SOURCE
	endif

	# query processor architecture
	UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)
		CFLAGS += -D AMD64
		CUDAFLAGS += -D AMD64
	else
		CFLAGS += -D x86
	endif

	# test gcc version
	VERSION5 := $(shell GCC_VERSION=$$(gcc -dumpversion); \
		[[ $$GCC_VERSION < 5.0 ]]; \
		echo $$? )

	ifeq (${VERSION5}, 0)
		CFLAGS += -D _POSIX_C_SOURCE=199309L
	endif
endif


###########
# RECIPES #

# phony targets are for those that don't produce files, ie. they don't act themselves on files
# even if there are files with names as those phony targets specified here they will be ignored
.PHONY : all display_results clean

# build all
all_display : display_results all

all : serial serial_openmp mpi mpi_openmp cuda cuda_shared


# build serial program
serial_display : display_results serial

serial : gol_serial_openmp.c gol_serial_openmp_alt_1dArr.c
	@$(CC) $(CFLAGS) -o gol_serial gol_serial_openmp.c
	@$(CC) $(CFLAGS) -o gol_serial_alt gol_serial_openmp_alt_1dArr.c
	@echo "Serial build"

serial_openmp_display : display_results serial_openmp

serial_openmp : gol_serial_openmp.c gol_serial_openmp_alt_1dArr.c
	$(eval CFLAGS += -fopenmp)
	@$(CC) $(CFLAGS) -o gol_serial_openmp gol_serial_openmp.c
	@$(CC) $(CFLAGS) -o gol_serial_alt_openmp gol_serial_openmp_alt_1dArr.c
	@echo "Serial OpenMP build"


# build CPU parallel 
# [gcc -Wall displays false positive warnings of variables not being used with MPI 
# (they are used internally by MPI)]
mpi_display : display_results mpi

mpi : gol_mpi_openmp.c
	@mpicc $(CFLAGS) -o gol_mpi gol_mpi_openmp.c
	@echo "MPI build"

mpi_openmp_display : display_results mpi_openmp

mpi_openmp : gol_mpi_openmp.c
	$(eval CFLAGS += -fopenmp)
	@mpicc $(CFLAGS) -o gol_mpi_openmp gol_mpi_openmp.c
	@echo "MPI + OpenMP build"


# cuda NVIDIA GPU build
cuda_display : display_results cuda

cuda : gol_cuda.cu
	@nvcc $(CUDAFLAGS) -o gol_cuda gol_cuda.cu
	@echo "CUDA build"

cuda_shared : gol_cuda.cu
	$(eval CUDAFLAGS += -D CUDA_SHARED)
	@nvcc $(CUDAFLAGS) -o gol_cuda_shared gol_cuda.cu
	@echo "CUDA Shared memory build"


# utilities
display_results : 
	$(eval CFLAGS += -D DISPLAY_RESULTS)
	$(eval CUDAFLAGS += -D DISPLAY_RESULTS)

clean :
	@rm -f $(EXECUTABLES) $(OBJECTS)
