#!/bin/bash

dim_hidden=16
ctx_wt=.5


for subj_id in {0..2}
do
   for lr in 1e-3 1e-4
   do
       for update_freq in 1 4 16
       do
           for dim_context in 64 256
           do
               for penalty_new_context in .1 .5 1
               do
                   for stickiness in .1 .5 1
                   do
                       sbatch train.sh
                           --${subj_id} \
                           --${lr} \
                           --${update_freq} \
                           --${dim_hidden} \
                           --${dim_context} \
                           --${ctx_wt} \
                           --${penalty_new_context} \
                           --${stickiness}
                   done
               done
           done
       done
   done
done
