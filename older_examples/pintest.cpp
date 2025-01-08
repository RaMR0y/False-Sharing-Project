#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define VAL_RANGE 1024

bool check(float *in, float *out, unsigned N) {
  for (int i = 0; i < N; i++)
    if (in[i] != out[i])
      return false;

  return true;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("usage: ./fs_bench1 buffer_size\n");
    exit(0);
  }

  unsigned N = atoi(argv[1]);
    
  float *A, *B;

  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);

  srand(time(0));
  for (int i = 0; i < N; i++)
    A[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  for (int i = 0; i < N; i++)
    B[i] = A[i];

  if (check(A, B, N))
    fprintf(stderr, "PASS\n");
  else
    fprintf(stderr, "FAIL\n");
  return 0;
}
