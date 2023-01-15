#!/bin/bash
#SBATCH -t 23:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 8G

#SBATCH --job-name=meta-run-cgru
#SBATCH --output slurm_log/%j.log

LOGROOT=/tigress/qlu/logs/meta-model/log
DT=$(date +%Y-%m-%d)

echo $(date)

srun python -u train.py \
    --subj_id ${1} \
    --lr ${2} \
    --update_freq ${3} \
    --dim_hidden ${4} \
    --dim_context ${5} \
    --ctx_wt ${6} \
    --stickiness ${7} \
    --lik_softmax_beta ${8} \
    --try_reset_h ${9} \
    --use_shortcut ${10} \
    --gen_grad ${11} \
    --exp_name $DT \
    --log_root $LOGROOT

echo $(date)
