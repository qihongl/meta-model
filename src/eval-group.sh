#!/bin/bash
#SBATCH -t 3:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 8G

#SBATCH --job-name=meta-eval
#SBATCH --output slurm_log/%j.log

LOGROOT=/tigress/qlu/logs/meta-model/log
DT=2023-02-18

echo $(date)

srun python -u eval-group.py \
    --lr ${1} \
    --update_freq ${2} \
    --dim_hidden ${3} \
    --dim_context ${4} \
    --ctx_wt ${5} \
    --stickiness ${6} \
    --lik_softmax_beta ${7} \
    --try_reset_h ${8} \
    --use_shortcut ${9} \
    --gen_grad ${10} \
    --pe_tracker_size ${11} \
    --match_tracker_size ${12} \
    --n_pe_std ${13} \
    --exp_name $DT \
    --log_root $LOGROOT

echo $(date)
