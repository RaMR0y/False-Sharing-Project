CC=nvcc
CXX=g++

FS_BENCH0=fs_bench0
FS_BENCH1=fs_bench1
UMAP=umap
FS=fs
PINTEST=pintest


all: ${FS_BENCH0} ${FS_BENCH1} 



${PINTEST}: ${PINTEST}.cpp
	${CXX} -c -I. -m64 -g ${PINTEST}.cpp
	${CXX} -o ${PINTEST} -m64 ${PINTEST}.o

${FS_BENCH0}: ${FS_BENCH0}.cu ${UMAP}
	${CC} -c -I. -g ${FS_BENCH0}.cu
	${CC} -o ${FS_BENCH0} ${FS_BENCH0}.o umap.o

${FS_BENCH1}: ${FS_BENCH1}.cu ${UMAP}
	${CC} -c -I. -w -g ${FS_BENCH1}.cu
	${CC} -o ${FS_BENCH1} ${FS_BENCH1}.o umap.o

${UMAP}: umap.h umap.cpp
	${CXX} -c -I. -Wall umap.cpp

${FS}: ${FS}.cpp ${UMAP}
	${CXX} -c -I. -w -g ${FS}.cpp
	${CXX} -o ${FS} ${FS}.o umap.o
clean:
	rm -rf *.o ${FS_BENCH0} ${FS_BENCH1}
