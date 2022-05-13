//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_profiler_api.h>
#include "nccl.h"
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char *argv[]) {

    int size;

    if(argc==2){
        size = atoi(argv[1]);
    }
    else if (argc>2){
        printf("too many arguments\n");
        return 1;
    }
    else{
        printf("expected an argument\n");
        return 1;
    }

    ncclComm_t comms[4];

    // managing 4 devices
    int nDev = 4;
    int devs[4] = {0, 1, 2, 3};

    // allocating and initializing device buffers
    float **sendbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff = (float **) malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    NCCLCHECK(ncclGroupStart());


  //##########################
    cudaProfilerStart();
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclAllReduce((const void *) sendbuff[i],
                                (void *) recvbuff[i], size, ncclFloat, ncclSum,
                                comms[i], s[i]));
    }
    cudaProfilerStop();
  //##############################



    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    printf("Success \n");
    return 0;
}