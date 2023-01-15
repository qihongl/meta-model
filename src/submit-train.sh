#!/bin/bash

ctx_wt=.5
lik_softmax_beta=0.33

for subj_id in {0..2}
do
  for lr in 1e-3
  do
    for update_freq in 4 32
    do
      for dim_context in 256
      do
        for dim_hidden in 16
        do
          for stickiness in 2 3
          do
            for try_reset_h in 0 1
            do
              for use_shortcut in 1
              do
                for gen_grad in 3 5 7
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
                       ${try_reset_h} \
                       ${use_shortcut} \
                       ${gen_grad} \
                       ${exp_name}
                done
              done
            done
          done
        done
      done
    done
  done
done
