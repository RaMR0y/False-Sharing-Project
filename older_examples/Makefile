CC=nvcc
CP_BENCH=um_cp
FS_BENCH0=fs_bench0
FS_BENCH1=fs_bench1
UM_MAP=umap
FS=fs

all:copy ${FS_BENCH0} ${FS_BENCH1} 

copy: 
	${CC} -o ${CP_BENCH} -w -g copy.cu

${FS_BENCH0}: ${FS_BENCH0}.cu
	${CC} -o ${FS_BENCH0} -w -g ${FS_BENCH0}.cu

${FS_BENCH1}: ${FS_BENCH1}.cu ${UM_MAP}
	${CC} -c -I. -w -g ${FS_BENCH1}.cu
	${CC} -o ${FS_BENCH1} ${FS_BENCH1}.o umap.o

${UM_MAP}: umap.h umap.cpp
	g++ -c -I. -Wall umap.cpp

${FS}: ${FS}.cpp ${UM_MAP}
	${CC} -c -I. -w -g ${FS}.cpp
	${CC} -o ${FS} ${FS}.o umap.o
clean:
	rm -rf ${CP_BENCH} ${FS_BENCH0} ${FS_BENCH1}
