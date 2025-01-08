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
echo "Input dependence experiments for ${bench} ..."
echo "y parameter runs ... "

		rm -rf ${bench}_fs_input_dep.csv
		
		if [ ${bench} = "fir" ]; then 
				yval=4
				ub=512
		fi
		if [ ${bench} = "ep" ]; then 
				yval=2
				ub=32
		fi
		while [ ${yval} -le $ub ]; do
			if [ ${bench} = "fir" ]; then 
					args="-w 0 -y ${yval} -x 750 -q"
					if [ $verbose ]; then 
							echo "y = $yval"
					fi
			fi
			if [ ${bench} = "ep" ]; then 
					args="-w 0 -c -m ${yval} -x 8 -q"
					if [ $verbose ]; then 
							echo "y = $yval"
					fi
			fi
			pin -t ${MEM_TRACE_GEN} -- ${prog} --mem host ${args} 
			umap > ${bench}_fs_report.out
			case=`cat ${bench}_fs_report.out | grep "malicious false sharing cases:" | awk '{print $NF}'`
			count=`cat ${bench}_fs_report.out | grep "malicious false sharing occurrences" | awk '{print $NF}'`
			
			echo $yval,$case,$count >> ${bench}_fs_input_dep.csv
			
			yval=$((${yval} * 2))
		done
		
		if [ ${bench} = "fir" ]; then 
				xval=375
				ub=48000
		fi
		if [ ${bench} = "ep" ]; then 
				xval=8
				ub=128
		fi
		while [ ${xval} -le $ub ]; do
			if [ ${bench} = "fir" ]; then 
					args="-w 0 -y 64 -x $xval -q"
					if [ $verbose ]; then 
							echo "x = $xval"
					fi
			fi
			if [ ${bench} = "ep" ]; then 
					args="-w 0 -c -m 2 -x $xval -q"
					if [ $verbose ]; then 
							echo "y = $yval"
					fi
			fi
			pin -t ${MEM_TRACE_GEN} -- ${prog} --mem host ${args} 
			umap > ${bench}_fs_report.out
			case=`cat ${bench}_fs_report.out | grep "malicious false sharing cases:" | awk '{print $NF}'`
			count=`cat ${bench}_fs_report.out | grep "malicious false sharing occurrences" | awk '{print $NF}'`
			
			echo $xval,$case,$count >> ${bench}_fs_input_dep.csv
			
			xval=$((${xval} * 2))
		done

