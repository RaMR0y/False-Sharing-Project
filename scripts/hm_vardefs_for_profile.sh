#!/bin/bash

bench=$1
all=$2
largest=$3

[ "$bench" ] || { echo "No benchmark specified. Exiting $0 ..."; exit 0; }
[ "$all" ] || { all=false; }
[ "$largest" ] || { largest=false; } 

# these locations are HARD-CODED; must customize if your environment is different 
HM_DIR=${HOME}/hm-raptor
INPUTDIR_BASE=${HM_DIR}/data            # assumes datasets are in top-level directory

bench_src_dir=${HM_DIR}/src/${bench}
bench_dir=${HM_DIR}/build/managed/src/${bench}/cuda   # path to CUDA executable
prog=${bench_dir}/${bench}_cuda                       # name of executable 
input_dir=${INPUTDIR_BASE}/${bench}                   # input dir for this benchmark 



# extract kernel name; handle special cases; 
case ${bench} in
	be)
		kernel="BackgroundExtraction"
		;;
	bs)
		kernel=${bench}_cuda
		;;
	ep)
		kernel="Evaluate_Kernel"
		;;	
	hist)
		kernel=Histogram
		;;
	kmeans)
		kernel=${bench}_compute_cuda
		;;
	fir)
		kernel=${bench}_cuda
		;;
	*)
		kernel=${bench}_cuda
		;;
esac

# set input for single run
if [ $all = "false" ]; then 
		if [ $largest = "true" ]; then 
				case ${bench} in
					aes)
						inputs="64MB.data"
						;;
					be)
						inputs="1920x1080.mp4"
						;;
					bs)
						inputs="8388608"
						;;
					ep)
						inputs="65536"
						;;
					fir)
						inputs="65536"
						;;
					ga)
						inputs="65536_1024.data"
						;;
					hist)
						inputs="2097152"
						;;
					kmeans)
						inputs="100000_34.txt"
						;;
					knn)
						inputs="knn_input3.txt"
						;;
					pr)
						inputs="16384.data"
						;;
				esac
		else
				case ${bench} in
					aes)
						inputs="1MB.data"
						;;
					be)
						inputs="320x180.mp4"
						;;
					bs)
						inputs="131072"
						;;
					bst)
						inputs="131072"
						;;
					ep)
						inputs="1024"
						;;
					fir)
						inputs="1024"
						;;
					ga)
						inputs="1024_64.data"
						;;
					hist)
						inputs="65536"
						;;
					kmeans)
						inputs="100_34.txt"
						;;
					knn)
						inputs="knn_input2.txt"
						;;
					pr)
						inputs="1024.data"
						;;
				esac
		fi
# else we are running with all inputs 
else 
	case ${bench} in
		aes)
			inputs="1MB.data 2MB.data 4MB.data 8MB.data 16MB.data 32MB.data"
			;;
		be)
			inputs="1024x576.mp4 1120x630.mp4 1280x720.mp4 1440x810.mp4 1600x900.mp4 1760x990.mp4\
            1920x1080.mp4 320x180.mp4 480x270.mp4 640x360.mp4 800x450.mp4 960x540.mp4"
			;;
		bs)
			# only varying number of elements; GPU chunks fixed at 4096
			inputs="131072 262144 524288 1048576 2097152 4194304 8388608"
			;;
		ep)
			# only varying number of blocks per kernel, number of kernel launches fixed at 1024 
			inputs="1024 2048 4096 8192 16384 32768 65536"
			;;
		fir)
			# only varying number of blocks per kernel, number of kernel launches fixed at 1024 
			inputs="1024 2048 4096 8192 16384 32768 65536"
			;;
		ga)
			inputs="1024_64.data 2048_128.data 4096_256.data 8192_512.data 16384_1024.data\
          								  32768_1024.data 65536_1024.data"
			;;
		hist)
			inputs="65536 131072 262144 524288 1048576 2097152"
			;;
		kmeans)
			inputs="100_34.txt 1000_34.txt 10000_34.txt 100000_34.txt 1000000_34.txt"
			;;
		knn)
			inputs="knn_input0.txt knn_input1.txt knn_input2.txt knn_input3.txt"
			;;
		pr)
			inputs="1024.data 2048.data 4096.data 8192.data 16384.data"
			;;
	esac
fi

for input in ${inputs}; do
	case ${bench} in
		aes)
			args="-w 0 -i ${input_dir}/$input -k ${input_dir}/key.data -q"
			;;
		be)
			args="-w 0 -g -i ${input_dir}/$input -q"
			;;
		bs)
			args="-w 0 -c -x $input -q"
			;;
		bst)
			args="-w 0 -i $input -q"
			;;
		ep)
			args="-w 0 -c -m 1600 -x 64 -q"
			;;
		fir)	 
			args="-w 0 -y 512 -x 750 -q"
			;;
		ga)
			args="-w 0 -c -i ${input_dir}/8192_512.data -q"
			;;
		hist)
			args="-w 0 -x $input -q "
			;;
		kmeans)
			args="-w 0 --min 1 -i ${input_dir}/10000_34.txt -q" 
			;;
		knn)
			args="-w 0 -i ${input_dir}/$input -q"
			;;
		pr)
			args="-w 0 -m 256 -i ${input_dir}/$input -q"
			;;
		*)
			args="-w 0 -i ${input_dir}/$input -q"
			;;
	esac
done

