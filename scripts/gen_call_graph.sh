#!/bin/bash
function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./gen_call_graph.sh [ OPTIONS ] bench

   generate call graph for Hetero-Mark benchmark 
	 call graph is written to cg.out 
   name of functions to be instrumented is written to instrument_funcs.out

Options: 
   --help    print this help message
   -a        run benchmark with all available input sets 

Optionss with values:

   -b <bench>      <bench> is a Hetero-Mark CUDA benchmark name
   
Examples:

   ./gen_call_graph.sh bs 

EOF
	exit 1
}

if [ "$1" = "--help" ]; then
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
    -a|--all)
      all=true
      ;;
    --largest)
      largest=true
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
[ "$all" ] || { all=false; }
[ "$largest" ] || { largest=false; }

if [ $DEBUG ]; then
		echo $prog
		echo ${input_dir}
		exit 0
fi

# set up environement 
CGEN=$HOME/pin/source/tools/ManualExamples/obj-intel64/call_graph.so
RAPTOR_SCRIPT_DIR=`dirname $0`

source ${RAPTOR_SCRIPT_DIR}/hm_vardefs.sh $bench $all $largest

if ! [ -e ${prog} ]; then
	echo "did not find executable ${prog}. attempting to build"
	pushd ${bench_dir}
	make
	pushd
	if [ -e ${prog} ]; then
			echo "build successful" 
	else
		echo "failed to build ${bench}, exiting ..."
	fi
fi

[ -r functions.in ] || { echo "functions.in not found. Exiting ..."; exit 0; }
echo "Generating call graph for ${bench} ..."
if [ $verbose ]; then 
		echo "pin -t ${CGEN} -- ${prog} --mem host ${args}"
fi
pin -t ${CGEN} -- ${prog} --mem host ${args}
