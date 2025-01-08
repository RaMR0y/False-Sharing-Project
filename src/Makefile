CXX=g++

UMAP_LIB_DIR=${RAPTOR_PATH}/lib
UMAP_BIN_DIR=${RAPTOR_PATH}/bin

UMAP=${UMAP_BIN_DIR}/umap
UMAPLIB=${UMAP_LIB_DIR}/umap.o

all: ${UMAP} ${UMAPLIB}

${UMAPLIB}: umap.h umap.cpp
	${CXX} -c -I. -Wall -o ${UMAPLIB} umap.cpp

${UMAP}: main.cpp ${UMAPLIB}
	${CXX} -c -I. -w -g main.cpp
	${CXX} -o ${UMAP} main.o ${UMAPLIB}

clean:
	rm -rf *.o ${UMAP} ${UMAPLIB} 
