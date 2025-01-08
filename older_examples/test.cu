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

void do_math(int *x) {
  *x += 5;
}

int main(int argc, char* argv[]) {
  bool host_alloc = false;
  if (argc < 2) {
    printf("usage: ./fs_demo N\n");
    exit(0);
  }

  // user specifies data set size in MB 
  // calculate number of floats to match data set size 
  unsigned bytes = atoi(argv[1]) * MB;
  unsigned int N = bytes/sizeof(float);

    
  float *A, *B;
  float *out, *dev_out;

  /* managed space */
  cudaMallocManaged(&A, sizeof(float) * N);
  cudaMallocManaged(&B, sizeof(float) * N);

  out = (float *) malloc(sizeof(float) * N);
  cudaMalloc((void **) &dev_out, sizeof(float) * N);
  
  srand(time(0));
  for (int i = 0; i < N; i++)
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  copy<<<N/BLOCKSIZE,BLOCKSIZE>>>(A, dev_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  if (check(A, out, N))
    printf("PASS\n");
  else
    printf("FAIL\n");

  return 0;
}
