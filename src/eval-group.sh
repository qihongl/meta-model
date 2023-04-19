#!/bin/bash
#SBATCH -t 3:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 8G

#SBATCH --job-name=meta-eval
#SBATCH --output slurm_log/%j.log

# LOGROOT=/tigress/qlu/logs/meta-model/log
LOGROOT=/scratch/gpfs/qlu/logs/meta-model/log
DT=2023-04-09

echo $(date)

srun python -u eval-group.py \
    --lr ${1} \
    --update_freq ${2} \
    --dim_hidden ${3} \
    --dim_context ${4} \
    --ctx_wt ${5} \
    --stickiness ${6} \
    --concentration ${7} \
    --lik_softmax_beta ${8} \
    --try_reset_h ${9} \
    --use_shortcut ${10} \
    --gen_grad ${11} \
    --pe_tracker_size ${12} \
    --match_tracker_size ${13} \
    --n_pe_std ${14} \
    --exp_name $DT \
    --log_root $LOGROOT

echo $(date)
