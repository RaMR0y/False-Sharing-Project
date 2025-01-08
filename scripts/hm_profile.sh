#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
Usage: ./run_hmark.sh [ OPTIONS ] 

   profile Hetero-Mark applications with different data placement configurations. 
   profile is written to ${mem}_raw.csv, where ${mem} is placement

Options: 
   --help    print this help message
   -p        profile benchmark 
   -v        verify results 
   -a        run benchmark with all available input sets 

Optionss with values:

   -b <bench>      <bench> is a Hetero-Mark executable
   -r <reps>       <reps> is number of repeated runs
   -m <mem>        data placement location: host, in, out, select or dev; default value is dev.
   
Examples:

   ./run_hmark.sh -b aes -m host -v  // run aes with host placement and verify results 

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
    -a|--all)
      all="true"
			;;
    -v|--verify)
      verify="true"
			;;
		--verbose)
			verbose="true"
			;;
    --pad)
      pad="true"
			;;
    -p|--profile)
      profile="true"
			;;
    -k|--keep)
      keep=1
			;;
    -r|--reps)
      reps="$2"
      shift 
			;;
    -m|--mem)
      mem="$2"
      shift 
			;;
    --largest)
      largest="true"
			;;
    *)
			echo "Unknown option:" $key
			exit 0
			;;
  esac
  shift
done

[ "$bench" ] || { echo "No benchmark specified. Exiting..."; exit 0; }
[ "$mem" ] || { mem="host"; }
[ "$all" ] || { all=false; }
[ "$largest" ] || { largest=false; }
[ "$pad" ] || { pad=false; }

[ "$reps" ] || { reps=1; }
[ "$verify" ] || { verify=false; }
[ "$profile" ] || { profile=false; }
[ "$verbose" ] || { verbose=false; }

# set up environement 
MEM_TRACE_GEN=$HOME/pin/source/tools/ManualExamples/obj-intel64/fs_trace.so
RAPTOR_SCRIPT_DIR=`dirname $0`

stats_script=${RAPTOR_SCRIPT_DIR}/average.py

source ${RAPTOR_SCRIPT_DIR}/hm_vardefs_for_profile.sh $bench $all $largest

if [ $pad == "true" ]; then 
	outfile=${bench}_fs_perf_pad.csv                                  # output file name 
else
	outfile=${bench}_fs_perf.csv                                  # output file name 
fi


if [ $DEBUG ]; then
		echo $prog
		echo $kernel
		echo ${input_dir}
		echo ${outfile}
		exit 0
fi

function convert_to_KB() {
	data=$1
	units="KB MB GB B"
	byte_convert_factor[0]="1"
	byte_convert_factor[1]="1024"
	byte_convert_factor[2]="1048576"
	byte_convert_factor[3]="1024"

	i=0
	for u in $units; do
	  if [ `echo $data | grep "$u"` ]; then
			if [ $i -eq 3 ]; then 
				data=`echo $data | sed 's/$u//' | awk -v val="${byte_convert_factor[$i]}" '{printf "%3.2f", $1 / val}'`  
			else
				data=`echo $data | sed 's/$u//' | awk -v val="${byte_convert_factor[$i]}" '{printf "%3.2f", $1 * val}'`  
			fi
			break
		fi 
		i=$(($i+1))
	done

	echo ${data}
}	


function profile() {

	input=$1
	if [ $verbose == "true" ]; then
		echo "${prog} --mem $mem ${args}" 
	fi
	${prog} --mem $mem --timing ${args} 2> runtime.dump

	run_time=`cat runtime.dump | grep "Run: " | awk '{print $2 * 1000}'`
	init_time=`cat runtime.dump | grep "Initialize: " | awk '{print $2 * 1000}'`
	
	# profile run 
	nvprof -u ms --system-profiling on --csv ${prog} --mem $mem ${args}  2> out.dump

	metric_search_str[0]="${kernel}("           # kernel execution time 
	metric_search_str[1]="Host To"              # H2D transfer time 
	metric_search_str[2]="Device To"            # D2H transfer time 
	metric_search_str[3]="Gpu"                  # GPU page fault time 
	metric_search_str[4]="Page throttles"       # GPU page throttling time 
	metric_search_str[5]="Memory thrashes"      # Memory thrashing time 

	metric_search_str[6]="Gpu"                  # GPU page fault count 
	metric_search_str[7]="Page throttles"       # Page throttle count 
	metric_search_str[8]="Memory thrashes"      # Memory thrashing count 

	metric_search_str[9]="CPU Page"             # CPU page faults 
	metric_search_str[10]="CPU thrashes"        # CPU thrashes 
	metric_search_str[11]="CPU throttles"       # CPU throttles 

	metric_search_str[12]="Host To"             # Number of H2D transfers 
	metric_search_str[13]="Host To"             # H2D avg transer size 
	metric_search_str[14]="Host To"             # H2D max transer size 
	metric_search_str[15]="Host To"             # H2D total transfer 

	metric_search_str[16]="Device To"           # Number of D2H transfers
	metric_search_str[17]="Device To"           # D2H avg transfer size 
	metric_search_str[18]="Device To"           # D2H max transfer size 
	metric_search_str[19]="Device To"           # D2H total transfer 
	
	metric_search_str[20]="Memory thrashes"     # Memory thrashing total transfer 

	metric_field[0]=3
	metric_field[1]=7
	metric_field[2]=7
	metric_field[3]=7
	metric_field[4]=7
	metric_field[5]=7
	metric_field[6]=2
	metric_field[7]=2
	metric_field[8]=2

	metric_field[9]=2
	metric_field[10]=2
	metric_field[11]=2

	metric_field[12]=2
	metric_field[13]=3
	metric_field[14]=5
	metric_field[15]=6

	metric_field[16]=2
	metric_field[17]=3
	metric_field[18]=5
	metric_field[19]=6

	metric_field[20]=6
	
	for i in {0..20}; do
		val=`cat out.dump | grep "${metric_search_str[$i]}"`
		if [ "val" ]; then
				if [ $i -eq 0 ]; then
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{if ($1 == "\"GPU activities\"") printf "%3.2f", $arg}'`
				fi
				if [ $i -ge 1 ] && [ $i -lt 6 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{printf "%3.2f", $arg}'`
						metrics[$i]=`echo ${metrics[$i]} | sed 's/ms//g'`
				fi
				if [ $i -ge 6 ] && [ $i -lt 9 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{printf "%d", $arg}'`
				fi
				if [ $i -ge 9 ] && [ $i -lt 12 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F ":" '{printf "%d", $arg}'`
						metrics[$i]=`echo ${metrics[$i]} | sed 's/ //g'`
				fi

				if [ $i -eq 12 ] || [ $i -eq 16 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{printf "%d", $arg}'`
				fi

				if [ $i -ge 13 ] && [ $i -lt 16 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{print $arg}'`
						result=$(convert_to_KB ${metrics[$i]})
            metrics[$i]=`echo $result | awk '{printf "%3.2f", $1}'`
				fi
				if [ $i -ge 17 ] && [ $i -lt 20 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{print $arg}'`
						result=$(convert_to_KB ${metrics[$i]})
            metrics[$i]=`echo $result | awk '{printf "%3.2f", $1}'`
				fi
				if [ $i -eq 20 ]; then 
						metrics[$i]=`echo $val | awk -v arg=${metric_field[$i]} -F "," '{print $arg}'`
						result=$(convert_to_KB ${metrics[$i]})
            metrics[$i]=`echo $result | awk '{printf "%3.2f", $1}'`
				fi
		else
					metrics[$i]=0.0
		fi
	done	

	if [ $largest == "true" ]; then 
			echo -n $bench",largest,"${init_time}","${run_time}","  >> raw_${outfile}
	else
		echo -n $bench",default,"${init_time}","${run_time}","  >> raw_${outfile}
	fi

	for i in {0..20}; do
		if [ $i -eq 20 ]; then 
				echo  ${metrics[$i]} >> raw_${outfile}
		else
				echo -n ${metrics[$i]}"," >> raw_${outfile}
		fi
	done

	# cleanup
	if [ ! "$keep" ]; then 
			rm -rf out.dump runtime.dump
	fi
}

#
# args for post-processing script
#

stat=avg
labels=2
cols=23
reps=$reps

j=0
if [ $profile = "true"  ]; then
	while [ $j -lt ${reps} ]; do
		echo "Profiling "$bench "... "
		profile 
		j=$(($j + 1))
	done
	python ${stats_script} raw_${outfile} $stat $labels $cols $reps > ${outfile}
else
	while [ $j -lt ${reps} ]; do
		if [ $verify ]; then 
				${prog} --mem $mem ${args} -v 
		else
			${prog} --mem $mem ${args} 
		fi
		j=$(($j + 1))
	done
fi

if [ ! "$keep" ]; then 
		rm -rf raw_${outfile}
fi
