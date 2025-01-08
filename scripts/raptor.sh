#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./raptor.sh [ OPTIONS ]

   Raptor driver script 

Options: 
   --help    print this help message

Optionss with values:

   -b <bench>      <bench> is a Hetero-Mark benchmark name 
   
Examples:

   ./raptor.sh -b bs   // run raptor on bs (Blackscholes)  

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

[ `which umap` ] || { echo "umap not in path. Exiting ..."; exit 0; }

RAPTOR_SCRIPT_DIR=`dirname $0`
source ${RAPTOR_SCRIPT_DIR}/hm_vardefs.sh $bench $all $largest

if [ "$verbose" == true ]; then
		verbose_flag="-v"
fi

#
# Extract program-defined function names for call-graph generation
#
[ -x ${RAPTOR_SCRIPT_DIR}/extract_func_names.sh ] || { echo "did not find extract_func_names.sh"; exit 0; }
${RAPTOR_SCRIPT_DIR}/extract_func_names.sh ${verbose_flag} ${bench}

[ -r ${bench}_functions.in ] || { echo "failed to extract function names from source"; exit 0; }
cp ${bench}_functions.in functions.in

#
# Generate call graph
#
[ -x ${RAPTOR_SCRIPT_DIR}/gen_call_graph.sh ] || { echo "did not find gen_call_graph.sh"; exit 0; }
${RAPTOR_SCRIPT_DIR}/gen_call_graph.sh ${verbose_flag} ${bench} # ${all} ${largest}
[ -r cg.out ] || { echo "failed to generate call graph"; exit 0; }
[ -r instrument_funcs.out ] || { echo "failed to generate list of functions to be instrumented"; exit 0; }
cp cg.out ${bench}_cg.out
cp instrument_funcs.out ${bench}_instrument_funcs.out

#
# Instrument source with calls to Raptor runtime
#

[ -x ${RAPTOR_SCRIPT_DIR}/add_src_instrumentation.sh ] || { echo "did not find add_src_instrumentation.sh"; exit 0; }
${RAPTOR_SCRIPT_DIR}/add_src_instrumentation.sh ${verbose_flag} ${bench}

#
# Build executable and link to Raptor runtime
# 
pushd ${bench_dir}
make 2> /dev/null
pushd 2> /dev/null

#
# Collect CPU memory trace and generate UM map
#

[ -x ${RAPTOR_SCRIPT_DIR}/gen_trace_and_umap.sh ] || { echo "did not find gen_trace_and_map.sh"; exit 0; }
${RAPTOR_SCRIPT_DIR}/gen_trace_and_umap.sh ${verbose_flag} ${bench} # ${all} ${largest}

#
# Analyze UM map and memory trace and generate False Sharing report
umap


