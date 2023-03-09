#!/bin/sh
export PYTHONPATH=./

PYTHON=python
dataset=$1
exp_name=$2
config=$3
epoch=$4


exp_dir=exp/${dataset}/${exp_name}

model_log=${exp_dir}/log
mkdir -p ${model_log}

save_folder=${exp_dir}/test_results/epoch_${epoch}
mkdir -p ${save_folder}

model_path=${exp_dir}/saved_model

now=$(date +"%Y%m%d_%H%M%S")

cp tool/sh_test.sh tool/test.py ${config} ${exp_dir}
$PYTHON tool/test.py \
    --config ${config} \
    --model_path ${model_path} \
    --epoch ${epoch} \
    --save_folder ${save_folder} 2>&1 | tee ${model_log}/test-$epoch-$now.log
