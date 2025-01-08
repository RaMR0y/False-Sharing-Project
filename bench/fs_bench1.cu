/* 
 * multiple managed spaces accessed by the kernel
 */
#include<stdio.h>
#include<time.h>

#define VAL_RANGE 1024
#define BLOCKSIZE 256

__global__ void mult(float *in1, float* in2, float *out) { 			  
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  out[tidx] = in1[tidx] * in2[tidx];
  return;
}

bool check(float *in1, float *in2, float *out, unsigned N) {
  float *result = (float *) malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++)
    result[i] = in1[i] * in2[i];

  for (int i = 0; i < N; i++)
    if (result[i] != out[i])
      return false;
  return true;
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    printf("usage: ./fs_bench0 <buffer_size>\n");
    exit(0);
  }
  
  unsigned int N = atoi(argv[1]);

  /* potential false sharing between B and C */
  float *A, *B, *C;

  float *out, *dev_out;

  cudaMallocManaged(&A, sizeof(float) * N);
  cudaMallocManaged(&B, sizeof(float) * N);

  out = (float *) malloc(sizeof(float) * N);
  cudaMalloc((void **) &dev_out, sizeof(float) * N);
  
  srand(time(0));
  for (int i = 0; i < N; i++) {
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
    B[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
  }
  unsigned grid_size = (N % BLOCKSIZE ? (N/BLOCKSIZE + 1) : N/BLOCKSIZE);
  mult<<<grid_size,BLOCKSIZE>>>(A, B, dev_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  cudaMallocManaged(&C, sizeof(float) * N);
  for (int i = 0; i < N; i++)
    C[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

#if 0  
  if (check(A, B, out, N))
    printf("PASS\n");
  else
    printf("FAIL\n");
#endif
  return 0;
}

