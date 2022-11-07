set -e -u

mode=$1 

# args
lr=1e-2
# arch=ViT-B/16
arch=blip
batch_size=128
job_name=${arch//\//\-}_lr${lr}xbs${batch_size}_noBias

if [ $mode = debug ]; then
	out_path=/media/song/myNull
elif [ $mode = exp ]; then
	out_path=/media/song/iaa/${job_name}/
	if [ -d $out_path ]; then
		echo Error: This experiment has already been implemented before. \
			See $out_path.
		exit 1
	fi
else
	echo "Unknown mode: $1"
	exit 1
fi

python -u main.py \
	--log-level INFO \
	--log-freq 1 \
	--epochs 100 \
	--arch $arch \
	--lr $lr \
	--data-root /home/song/AVA/ \
	--batchSize $batch_size \
	--num-workers 10 \
	--device cuda:0 \
	--out-path $out_path \
	# --freeze-feats
