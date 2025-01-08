#include<stdio.h>
#include<time.h>

#define VAL_RANGE 1024
#define BLOCKSIZE 1024
#define MB 1048576

__global__ void copy(float *in, float *out) { 			  
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  out[tidx] = in[tidx];
  return;
}

bool check(float *in, float *out, unsigned N) {
  for (int i = 0; i < N; i++)
    if (in[i] != out[i])
      return false;
  return true;
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    printf("usage: ./fs_bench0 <allocation_in_bytes>\n");
    exit(0);
  }
  
  unsigned bytes = atoi(argv[1]) * MB;
  unsigned int N = bytes/sizeof(float);
    
  float *A;
  float *B, *dev_B;

  cudaMallocManaged(&A, sizeof(float) * N);

  B = (float *) malloc(sizeof(float) * N);
  cudaMalloc((void **) &dev_B, sizeof(float) * N);
  
  srand(time(0));
  for (int i = 0; i < N; i++)
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  copy<<<N/BLOCKSIZE,BLOCKSIZE>>>(A, dev_B);
  cudaDeviceSynchronize();

  cudaMemcpy(B, dev_B, sizeof(float) * N, cudaMemcpyDeviceToHost);

  if (check(A, B, N))
    printf("PASS\n");
  else
    printf("FAIL\n");

  return 0;
}

