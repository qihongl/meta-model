#!/bin/bash
dim_hidden=16
ctx_wt=.5
penalty_new_context=0
lik_softmax_beta=0.333

for subj_id in {0..2}
do
   for lr in 1e-3 1e-4
   do
       for update_freq in 1 4 16
       do
           for dim_context in 64 256
           do
               for stickiness in 1 4
               do
                   sbatch train.sh \
                       ${subj_id} \
                       ${lr} \
                       ${update_freq} \
                       ${dim_hidden} \
                       ${dim_context} \
                       ${ctx_wt} \
                       ${penalty_new_context} \
                       ${stickiness} \
                       ${lik_softmax_beta}
               done
           done
       done
   done
done
