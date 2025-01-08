#include<stdio.h>
#include<stdlib.h>
#include<time.h>

//**** BEGIN CODE INSERTED BY AUTOFS  ****/ 
#include<umap.h>
um_map map; 
unsigned alloc_id = 0;
/**** END CODE INSERTED BY AUTOFS  ****/ 

#define VAL_RANGE 1024
#define BLOCKSIZE 256
#define MB 1048576

//#define DEBUG 1

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
    printf("usage: ./fs_bench1 buffer_size\n");
    exit(0);
  }

  unsigned N = atoi(argv[1]);
    
  float *A, *B;
  float *out, *dev_out;

  /* managed space */
  cudaMallocManaged(&A, sizeof(float) * N);

  /**** BEGIN CODE INSERTED BY AUTOFS  ****/ 
  float *begin_A = A;      
  unsigned long __size = sizeof(float) * N; 
  /* update virtual map */ 
  map.update(alloc_id, begin_A, __size); 
  alloc_id++;
  /**** END CODE INSERTED BY AUTOFS  ****/ 

#ifdef DEBUG
  unsigned long adjusted_end_A = ((unsigned long) end_A) - 1;
  unsigned long size = (unsigned long) end_A - (unsigned long) begin_A;
  printf("[A] begin address: %p\n", begin_A);
  printf("[A] end address: %p\n", (void *) adjusted_end_A);
  printf("[A] size: %lu bytes\n", size);
  printf("[A] UM pages: %lu\n", size/BASE_PAGE_SIZE);
#endif

  out = (float *) malloc(sizeof(float) * N);
  cudaMalloc((void **) &dev_out, sizeof(float) * N);
  
  srand(time(0));

  for (int i = 0; i < N; i++)
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);


  unsigned grid_size = (N/BLOCKSIZE == 0 ? N/BLOCKSIZE : (N/BLOCKSIZE + 1));
  copy<<<grid_size,BLOCKSIZE>>>(A, dev_out);

  /**** BEGIN CODE INSERTED BY AUTOFS  ****/ 
  map.update_gpu_access(0);  
  /**** END CODE INSERTED BY AUTOFS  ****/ 

  cudaDeviceSynchronize();
  cudaMemcpy(out, dev_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
  
  cudaMallocManaged(&B, sizeof(float) * N);

  /**** BEGIN CODE INSERTED BY AUTOFS  ****/ 
  float *begin = B;      
  __size = sizeof(float) * N; 
  map.update(alloc_id, begin, __size);   /* update UM map */ 
  alloc_id++;
  /**** END CODE INSERTED BY AUTOFS  ****/ 

#ifdef DEBUG
  unsigned long adjusted_end = ((unsigned long) end) - 1;
  size = (unsigned long) end - (unsigned long) begin;
  printf("[B] begin address: %p\n", begin);
  printf("[B] end address: %p\n", (void *) adjusted_end);
  printf("[B] size: %lu bytes\n", size);
  printf("[B] UM pages: %lu\n", size/BASE_PAGE_SIZE);
#endif
  
  // host access of B
  for (int i = 0; i < N; i++)
    B[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

#if DEBUG     
  if (check(A, out, N))
    fprintf(stderr, "PASS\n");
  else
    fprintf(stderr, "FAIL\n");
#endif
  map.dump(false);
  return 0;
}
