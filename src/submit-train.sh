#!/bin/bash

ctx_wt=.5
lik_softmax_beta=0.33

for subj_id in {0..2}
do
  for lr in 1e-3
  do
    for update_freq in 1 4
    do
      for dim_context in 64 256
      do
        for dim_hidden in 16
        do
          for stickiness in 1 1.5 2
          do
            for try_reset_h in 0 1
            do
              sbatch train.sh \
                   ${subj_id} \
                   ${lr} \
                   ${update_freq} \
                   ${dim_hidden} \
                   ${dim_context} \
                   ${ctx_wt} \
                   ${stickiness} \
                   ${lik_softmax_beta} \
                   ${try_reset_h}
            done
          done
        done
      done
    done
  done
done
