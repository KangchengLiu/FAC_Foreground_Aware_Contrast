#!/bin/sh
export PYTHONPATH=./

PYTHON=python
dataset=$1
exp_name=$2
config=$3

exp_dir=exp/${dataset}/${exp_name}

model_log=${exp_dir}/log
mkdir -p ${model_log}

model_path=${exp_dir}/saved_model
mkdir -p ${model_path}

model_events=${exp_dir}/events
mkdir -p ${model_events}

now=$(date +"%Y%m%d_%H%M%S")

cp tool/sh_train.sh tool/train.py ${config} ${exp_dir}

$PYTHON tool/train.py \
    --config ${config} \
    --model_path ${model_path} \
    --save_path=${exp_dir} 2>&1 | tee ${model_log}/train-$now.log

