#ifndef BYPASS_INTELLISENSE
#define BYPASS_INTELLISENSE

#ifdef __INTELLISENSE__
// put functions unrecognized by intellisense here - for use to solve bugs with MSVC
void __syncthreads();
#endif

#endif