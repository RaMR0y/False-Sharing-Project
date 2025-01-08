#!/bin/bash
function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./gen_trace_and_umap.sh [ OPTIONS ] bench

   generate memory trace of CPU access and UM map for Hetero-mark bench 
   memory trace is written to cpu_mem_trace.out
	 UM map is written to umap.out

Options: 
   --help    print this help message
   -a        run benchmark with all available input sets 

Optionss with values:

   -b <bench>      <bench> is a Hetero-Mark CUDA benchmark name
   
Examples:

   ./gen_trace_and_map.sh bs 

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
MEM_TRACE_GEN=$HOME/pin/source/tools/ManualExamples/obj-intel64/fs_trace.so
RAPTOR_SCRIPT_DIR=`dirname $0`

source ${RAPTOR_SCRIPT_DIR}/hm_vardefs.sh $bench $all $largest

[ -r instrument_funcs.out ] || { echo "instrument_funcs.out not found. Bailing ..."; exit 0; }
echo "Collecting CPU memory trace for ${bench} ..."
if [ $verbose ]; then 
		echo "pin -t ${MEM_TRACE_GEN} -- ${prog} --mem host ${args}"
fi
pin -t ${MEM_TRACE_GEN} -- ${prog} --mem host ${args} 
