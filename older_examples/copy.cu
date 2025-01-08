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
    printf("usage: ./um_cp N\n");
    exit(0);
  }
  // user specifies data set size in MB 
  // calculate number of floats to match data set size 
  unsigned bytes = atoi(argv[1]) * MB;
  unsigned int N = bytes/sizeof(float);
  printf("%d\n", N);
  if (argc > 2)
    host_alloc = (bool) atoi(argv[2]);
  else
    host_alloc = 0;
    
  float *in, *out;
  float *dev_in, *dev_out;

  if (host_alloc) {
    cudaMallocManaged(&in, sizeof(float) * N);
    cudaMallocManaged(&out, sizeof(float) * N);
  }
  else {
    in = (float *) malloc(sizeof(float) * N);
    cudaMalloc((void **) &dev_in, sizeof(float) * N);
    out = (float *) malloc(sizeof(float) * N);
    cudaMalloc((void **) &dev_out, sizeof(float) * N);
  }


  srand(time(0));

  for (int i = 0; i < N; i++)
    in[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
   

  if (!host_alloc)
    cudaMemcpy(dev_in, in, sizeof(float) * N, cudaMemcpyHostToDevice);

  if (!host_alloc) {
    copy<<<N/BLOCKSIZE,BLOCKSIZE>>>(dev_in, dev_out);
    cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
  }
  else {
    copy<<<N/BLOCKSIZE,BLOCKSIZE>>>(in, out);
    cudaDeviceSynchronize();
  }

#if 1
  if (check(in, out, N))
    printf("PASS\n");
  else
    printf("FAIL\n");
#endif 
  return 0;
}
