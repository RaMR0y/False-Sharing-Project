#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./extract_funct_names.sh [OPTIONS] <bench>

    script invokes LLVM libtool get-func-names libtool to extract 
    names of functions defined in a Hetero-Mark application. 
    get-func-name is applied to each source file. 
    Assumes Hetoro-Mark installation is at ${HOME}/hm

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

RAPTOR_SCRIPT_DIR=`dirname $0`
LLVM_PATH=$HOME/llvm/bin
source ${RAPTOR_SCRIPT_DIR}/hm_vardefs.sh $bench

srcs="${bench_src_dir}/${bench}_benchmark*.cc ${bench_src_dir}/cuda/*.cu ${bench_src_dir}/cuda/*.cc"

rm -rf ${bench}_functions.in
for f in $srcs; do
	[ -e $f ] || { echo "file not found $f"; exit 0; }
	if [ $verbose ]; then 
			echo $f
	fi
	ext=`echo $f | awk -F "." '{print $NF}'`
	if [ $ext == "cu" ]; then 
			${LLVM_PATH}/get-func-names $f  -- -w -I$HOME/hm --cuda-gpu-arch=sm_61 -pthread  -std=c++11 >> ${bench}_functions.in 
	else
		(${LLVM_PATH}/get-func-names $f  -- -w -I$HOME/hm  -pthread  -std=c++11 2>/dev/null) >> ${bench}_functions.in 
	fi
done
