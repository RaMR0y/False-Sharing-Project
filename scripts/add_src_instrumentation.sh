#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./add_src_instrumentation.sh [OPTIONS] <bench>

    adds source code instrumentation
    invokes LLVM libtool false-sharing-detect to inject calls to Raptor runtime. 
    invoked on CUDA source and main.cc
    expects to find *.raptor files in src directory  

Options: 
   <bench>           a Hetero-Mark benchmark name 

   --help           print this help message
   -v, --verbose    print diagnostic messages

EOF
	exit 1
}

if [ $# -lt 1 ] || [ $1 = "--help" ]; then
	usage
fi

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    -h|--help)
      usage 
			;;
    -v|--verbose)
      verbose=true 
			;;
    *)
			if [ $# -eq 1 ]; then 
				bench=$arg
			else
				echo "Unknown option:" $key
				exit 0
			fi
			;;
  esac
  shift
done

[ "$bench" ] || { echo "No benchmark specified. Exiting..."; exit 0; }

if [ $DEBUG ]; then
		echo $prog
		exit 0
fi

RAPTOR_SCRIPT_DIR=`dirname $0`
LLVM_PATH=$HOME/llvm/bin
source ${RAPTOR_SCRIPT_DIR}/hm_vardefs.sh $bench

srcs="${bench_src_dir}/cuda/*.cu.raptor ${bench_src_dir}/cuda/main.cc.raptor"

for f in $srcs; do
	[ -e $f ] || { echo "file not found $f"; exit 0; }
	if [ $verbose ]; then 
			echo $f
	fi

	# need three slashes to escape the dot once for the string and once for the shell (Lookup TODO)
	file_name=`echo $f | awk -F '\\\.raptor' '{print $1}'`
	ext=`echo ${file_name} | awk -F '.' '{print $NF}'`
	cp $f ${file_name}
									 
	if [ $ext == "cu" ]; then 
			if [ $verbose ]; then 
				echo "${LLVM_PATH}/false-sharing-detect ${file_name} -- -I$HOME/hm-raptor --cuda-gpu-arch=sm_70 -pthread -w -std=c++11 2> tmp"
			fi
			${LLVM_PATH}/false-sharing-detect ${file_name} -- -I$HOME/hm-raptor --cuda-path=/usr/local/cuda-10.0 --cuda-gpu-arch=sm_61 -pthread -w -std=c++11 2> tmp
	else
			if [ $verbose ]; then 
				echo "${LLVM_PATH}/false-sharing-detect ${file_name} -- -I$HOME/hm-raptor  -I/usr/local/cuda/include -w -pthread -std=c++11 2> tmp"
			fi
			${LLVM_PATH}/false-sharing-detect ${file_name} -- -I$HOME/hm-raptor  -I/usr/local/cuda/include -w -pthread -std=c++11 2> tmp
	fi
	mv tmp ${file_name}
done
