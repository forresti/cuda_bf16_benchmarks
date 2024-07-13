#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

double read_timer();

const char* _curandGetErrorString(curandStatus_t error);

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

#define CHECK_CURAND(x) do { \
  curandStatus_t res = (x); \
  if(res != CURAND_STATUS_SUCCESS) { \
    fprintf(stderr, "CURAND: %s = %d (%s) at (%s:%d)\n", #x, res, _curandGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)





