#!/bin/bash

echo Running on python environment: $CONDA_DEFAULT_ENV

finetuning_archive=
original_archive=
model_name_suffix=

while [ $# -gt 0 ] ; do
  case $1 in
    -f | --finetuning-archive) finetuning_archive="$2" ;;
    -o | --original-archive) original_archive="$2" ;;
    -m | --model-name-suffix) model_name_suffix="$2" ;;
  esac
  shift
done

if test -z "$finetuning_archive"; then
   echo Please provide the finetuning archive
   echo The name of the file should be 'donkey-<date>-finetuning-agent-dave2-m4-generated_track-0-run-<index>.npz'. The string to be provided should be 'donkey-<date>-finetuning-agent-dave2-m4-generated_track-0'
   exit 1
fi

if test -z "$original_archive"; then
   echo Please provide the original archive
   echo The name of the file should be 'donkey-archive-agent-autopilot-episodes-<num-episodes>-generated_track-0', which is also the string that should be provided.
   exit 1
fi

if test -z "$model_name_suffix"; then
   echo Please provide the model name suffix
   echo The name of the file should be the name of the model, i.e., 'donkey-dave2-m4', without extension
   exit 1
fi


directory=$PWD
cd ..

num_runs=$(find logs -name "*$finetuning_archive-run-*.npz" | wc -l)

for i in $num_runs; do 
    python train_model.py --seed 0 --env-name donkey --archive-names "$finetuning_archive"-run-$i.npz "$original_archive".npz \
      --model-name dave2 --test-split 0.2 --keep-probability 0.5 --learning-rate 1e-6 --nb-epoch 1000 --batch-size 128 \
      --early-stopping-patience 10 --percentage-data 0.5 \
      --model-name-suffix "$model_name_suffix"-run-$i > logs/models/"$model_name_suffix"-run-$i.txt
done

cd $directory




