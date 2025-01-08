#include<stdio.h>
#include<stdlib.h>
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
  printf("[A] begin address: %p\n", &A);
  printf("[A] end address: %p\n", &A + (sizeof(float) * N));
  printf("[A] number pages: %d\n", (sizeof(float) * N)/(4 * 1024));
  cudaMallocManaged(&B, sizeof(float) * N);
  printf("[B] begin address: %p\n", &B);
  printf("[B] end address: %p\n", &B + (sizeof(float) * N));
  printf("[B] number pages: %d\n", (sizeof(float) * N)/(4 * 1024));


  out = (float *) malloc(sizeof(float) * N);
  cudaMalloc((void **) &dev_out, sizeof(float) * N);
  
  srand(time(0));

  for (int i = 0; i < N; i++)
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  copy<<<N/BLOCKSIZE,BLOCKSIZE>>>(A, dev_out);
  cudaDeviceSynchronize();
  cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
  
  // host access of B
  for (int i = 0; i < N; i++)
    B[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
  
     
  if (check(A, out, N))
    printf("PASS\n");
  else
    printf("FAIL\n");

  return 0;
}
