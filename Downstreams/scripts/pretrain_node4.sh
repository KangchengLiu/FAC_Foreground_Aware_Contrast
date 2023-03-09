
#! /bin/bash
#SBATCH --job-name=FAC
#SBATCH --job-name=FAC
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --partition=dev
#SBATCH --comment="FAC"
#SBATCH --constraint=volta32gb

#SBATCH --signal=B:USR1@60
#SBATCH --open-mode=append

LOG_PATH=$2
mkdir -p $LOG_PATH

export PYTHONPATH=$PWD:$PYTHONPATH

srun --output=${LOG_PATH}/%j.out --error=${LOG_PATH}/%j.err --label python scripts/multinode-wrapper.py main.py $1
