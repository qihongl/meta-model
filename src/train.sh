#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 2G

#SBATCH --job-name=meta-run-cgru
#SBATCH --output slurm_log/%j.log

LOGROOT=/tigress/qlu/logs/meta-model/log
# LOGROOT=/scratch/gpfs/qlu/logs/meta-model/log

# DT=$(date +%Y-%m-%d)
DT=freeze-ctx-test

echo $(date)

srun python -u train.py \
    --subj_id ${1} \
    --lr ${2} \
    --update_freq ${3} \
    --dim_hidden ${4} \
    --dim_context ${5} \
    --ctx_wt ${6} \
    --stickiness ${7} \
    --concentration ${8} \
    --lik_softmax_beta ${9} \
    --try_reset_h ${10} \
    --use_shortcut ${11} \
    --gen_grad ${12} \
    --pe_tracker_size ${13} \
    --match_tracker_size ${14} \
    --n_pe_std ${15} \
    --exp_name $DT \
    --log_root $LOGROOT

echo $(date)
