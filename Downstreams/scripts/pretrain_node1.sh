
#! /bin/bash
#SBATCH --job-name=FAC
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --partition=dev
#SBATCH --comment="test"
#SBATCH --constraint=volta32gb

#SBATCH --signal=B:USR1@60
#SBATCH --open-mode=append

EXPERIMENT_PATH="./checkpoints/testlog"
mkdir -p $EXPERIMENT_PATH

export PYTHONPATH=$PWD:$PYTHONPATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python scripts/singlenode-wrapper.py main.py $1
