#!/bin/bash

subj_id=99
lr=1e-3
update_freq=100
dim_hidden=16
dim_context=16
ctx_wt=.5
penalty_new_context=.5
stickiness=.5

sbatch train.sh
    ${subj_id} \
    ${lr} \
    ${update_freq} \
    ${dim_hidden} \
    ${dim_context} \
    ${ctx_wt} \
    ${penalty_new_context} \
    ${stickiness}
