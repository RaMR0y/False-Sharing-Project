#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
Usage:  fs_test.sh [ OPTIONS ] -- bench [ bench args ]

Options: 
   --help          print this help message	 
   -b, --bench <benchmark>
EOF
	exit 1
}

function cleanup() {
	if [ ! "$keep" ]; then 
	  rm -rf tmp *.gen umap.out pinatrace.out ranges.in
	fi
}

if [ "$1" = "--help" ] || [ $# -lt 1 ]; then
	usage
fi

while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -b|--bench)
      bench="$2"
      shift 
      ;;
		--keep)
      keep=true
      ;;
    --)
      shift;
      bench_args="$@"
      break;
      ;;
    *)
      echo Unknown option: $key
			usage
      exit 0
      ;;
  esac
  shift 
done

src=$bench.cu

RAPTOR_BIN_PATH=${RAPTOR_PATH}/bin
LLVM_BIN_PATH=${HOME}/llvm/bin

S2S=${LLVM_BIN_PATH}/false-sharing-detect
S2S_FLAGS="--cuda-gpu-arch=sm_61"

PIN_TOOL=$PIN_OBJ_PATH/pinatrace.so
UMAP=${RAPTOR_BIN_PATH}/umap

cp ${RAPTOR_PATH}/bench/$src .
cp ${RAPTOR_PATH}/bench/Makefile .

${S2S} $src -- ${S2S_FLAGS} 2> $src.gen
cp $src.gen $src
make $bench > /dev/null

pin -t ${PIN_TOOL} -- ${bench_args} > umap.out
fs_report=`$UMAP`
# profile the code to get data migration numbers 
(nvprof -u ms  --system-profiling on ${bench_args} > /dev/null) 2> tmp
D2H=`cat tmp | grep "Device To" | awk '{printf "%2.0f", $5}'`
if [ "$D2H" = "" ]; then
		D2H=0
fi
H2D=`cat tmp | grep "Host To" | awk '{printf "%2.0f", $5}'`
echo ${fs_report},$H2D,$D2H

cleanup
