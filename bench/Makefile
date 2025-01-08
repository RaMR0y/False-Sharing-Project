NVCC=nvcc
CXX=g++

UMAP_LIB_DIR=${RAPTOR_PATH}/lib
UMAP_INC_DIR=${RAPTOR_PATH}/src

UMAP=${UMAP_BIN_DIR}/umap
UMAPLIB=${UMAP_LIB_DIR}/umap.o

FS_BENCH0=fs_bench0
FS_BENCH1=fs_bench1
FS_BENCH2=fs_bench2
FS_BENCH3=fs_bench3

all: ${FS_BENCH0} ${FS_BENCH1} ${FS_BENCH2} ${FS_BENCH3} 

${FS_BENCH0}: ${FS_BENCH0}.cu 
	${NVCC} -c -I${UMAP_INC_DIR} -w -g ${FS_BENCH0}.cu
	${NVCC} -o ${FS_BENCH0} ${FS_BENCH0}.o ${UMAPLIB}

${FS_BENCH1}: ${FS_BENCH1}.cu
	${NVCC} -c -I${UMAP_INC_DIR} -w -g ${FS_BENCH1}.cu
	${NVCC} -o ${FS_BENCH1} ${FS_BENCH1}.o ${UMAPLIB}

${FS_BENCH2}: ${FS_BENCH2}.cu
	${NVCC} -c -I${UMAP_INC_DIR} -w -g ${FS_BENCH2}.cu
	${NVCC} -o ${FS_BENCH2} ${FS_BENCH2}.o ${UMAPLIB}

${FS_BENCH3}: ${FS_BENCH3}.cu
	${NVCC} -c -I${UMAP_INC_DIR} -w -g ${FS_BENCH3}.cu
	${NVCC} -o ${FS_BENCH3} ${FS_BENCH3}.o ${UMAPLIB}

clean:
	rm -rf *.o  ${FS_BENCH0} ${FS_BENCH1} ${FS_BENCH2} ${FS_BENCH3}
