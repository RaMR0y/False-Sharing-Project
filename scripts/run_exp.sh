#!/bin/bash

begin_size=16128
#begin_size=16592
end_size=16640

echo "size,padding,FS_page_index,H2D_migration(KB),D2H_migration(KB)"
size=${begin_size}
while [ $size -le ${end_size} ]; do
	pin -t $PIN_OBJ_PATH/pinatrace.so -- ./fs_bench1 $size > um_map.txt
	cp pinatrace.out ${size}_trace_new.out
#	./fs_bench1 $size > um_map.txt
	fs_result=`./fs`
#	page=`cat tmp | grep "PAGE" | awk '{print $1}' | awk -F ":" '{print $2}' | sed 's/]//g'`
#	if [ "$page" = "" ]; then
#			page=-1
#	fi
#	padding=`cat tmp | grep  "\[ALLOC:0\] padding" | awk '{print $3}'`
	(nvprof -u ms  --system-profiling on ./fs_bench1 $size > /dev/null) 2> tmp
	D2H=`cat tmp | grep "Device To" | awk '{printf "%3.2f", $5}'`
	if [ "$D2H" = "" ]; then
			D2H=0
	fi
	H2D=`cat tmp | grep "Host To" | awk '{printf "%3.2f", $5}'`
#	echo $size,$padding,$page,$H2D,$D2H
	echo $size,${fs_result},$H2D,$D2H
	size=$(($size+16))
done

for size in 16383 16385; do

	# apply pintool to source-instrumented CUDA binary 
	pin -t $PIN_OBJ_PATH/pinatrace.so -- ./fs_bench1 $size > um_map.txt

	# save trace for later, if needed 
	cp pinatrace.out ${size}_trace_new.out

	# get_ false sharing results 
	fs_result=`./fs`

	# verify by collecting page migration data 

	(nvprof -u ms  --system-profiling on ./fs_bench1 $size > /dev/null) 2> tmp

	# device-to-host migration
	# in the microbenchmarks, we designed, device-to-host migration is 0 when there is no false sharing
	D2H=`cat tmp | grep "Device To" | awk '{printf "%3.2f", $5}'`
	if [ "$D2H" = "" ]; then
			D2H=0
	fi
	# host-to-device migration
	H2D=`cat tmp | grep "Host To" | awk '{printf "%3.2f", $5}'`
	echo $size,${fs_result},$H2D,$D2H
done


