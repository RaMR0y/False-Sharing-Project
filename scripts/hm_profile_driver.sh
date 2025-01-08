#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./hm_profile_driver.sh [ OPTIONS ] 

   driver for profiling baseline and padded versions of Hetero-Mark 

Options: 
   --help    print this help message
   --pad     only profile padded version     

Optionss with values:

   -b <bench>      <bench> is a Hetero-Mark executable
   -r <reps>       <reps> is number of repeated runs

Examples:

   ./hm_profile_driver.sh -b fir --pad --reps 5  // profile padded fir with 5 reps

EOF
	exit 1
}

if [ "$1" = "--help" ]; then
	usage
fi

while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -b|--bench)
      bench="$2"
      shift 
			;;
    --pad)
      pad_only="true"
			;;
    -r|--reps)
      reps="$2"
      shift 
			;;
    *)
			echo "Unknown option:" $key
			exit 0
			;;
  esac
  shift
done

[ "$bench" ] || { echo "No benchmark specified. Exiting..."; exit 0; }
[ "$reps" ] || { reps=1; }
[ "$pad_only" ] || { pad_only=false; }


HM_DIR=${HOME}/hm-raptor
RAPTOR_SCRIPT_DIR=`dirname $0`

profiler="${RAPTOR_SCRIPT_DIR}/hm_profile.sh"

bench_src_dir=${HM_DIR}/src/${bench}/cuda
bench_build_dir=${HM_DIR}/build/managed/src/${bench}   

cuda_src_file="${bench_src_dir}/${bench}_cuda_benchmark.cu"
main_file="${bench_src_dir}/main.cc"


#
# Baseline 
# 

if [ ${pad_only} == "false" ]; then 
	cp ${cuda_src_file}.raptor ${cuda_src_file}
	cp ${main_file}.raptor ${main_file}
	pushd ${bench_build_dir} &> /dev/null
	make
	popd &> /dev/null
	$profiler -b ${bench} --profile --keep --reps ${reps}
fi

#
# Padding
#
cp ${cuda_src_file}.pad ${cuda_src_file}
cp ${main_file}.raptor ${main_file}
pushd ${bench_build_dir} &> /dev/null
make
popd &> /dev/null

$profiler -b ${bench} --profile --pad --keep --reps ${reps}

