#include<stdio.h>
#include<time.h>

#define VAL_RANGE 1024
#define BLOCKSIZE 256

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
    printf("usage: ./fs_bench0 <buffer_size>\n");
    exit(0);
  }
  
  unsigned int N = atoi(argv[1]);

  /* potential false sharing between A and B */
  float *A, *B;

  float *out, *dev_out;

  cudaMallocManaged(&A, sizeof(float) * N);

  out = (float *) malloc(sizeof(float) * N);
  cudaMalloc((void **) &dev_out, sizeof(float) * N);
  
  srand(time(0));
  for (int i = 0; i < N; i++)
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  unsigned grid_size = (N % BLOCKSIZE ? (N/BLOCKSIZE + 1) : N/BLOCKSIZE);
  copy<<<grid_size,BLOCKSIZE>>>(A, dev_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  cudaMallocManaged(&B, sizeof(float) * N);
  for (int i = 0; i < N; i++)
    B[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

#if 0  
  if (check(A, out, N))
    printf("PASS\n");
  else
    printf("FAIL\n");
#endif
  return 0;
}

