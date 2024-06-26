#!/bin/bash

ctx_wt=.5
lik_softmax_beta=0.33

# gen_grad=.5
# pe_tracker_size=32
# match_tracker_size=4
# n_pe_std=3

for lr in 1e-3
do
  for update_freq in 4
  do
    for dim_context in 128
    do
      for dim_hidden in 16 32
      do
        for stickiness in 4
        do
          for concentration in 1 2
          do
            for try_reset_h in 0
            do
              use_shortcut=1
              for gen_grad in .15 .25 .35
                do
                  for pe_tracker_size in 256
                  do
                    for match_tracker_size in 4
                    do
                      for n_pe_std in .5 1 2
                      do
                      sbatch eval-group.sh \
                           ${subj_id} \
                           ${lr} \
                           ${update_freq} \
                           ${dim_hidden} \
                           ${dim_context} \
                           ${ctx_wt} \
                           ${stickiness} \
                           ${concentration} \
                           ${lik_softmax_beta} \
                           ${try_reset_h} \
                           ${use_shortcut} \
                           ${gen_grad} \
                           ${pe_tracker_size} \
                           ${match_tracker_size} \
                           ${n_pe_std} \
                           ${exp_name}
                     done
                   done
                 done
              done

              use_shortcut=0
              sbatch eval-group.sh \
                   ${subj_id} \
                   ${lr} \
                   ${update_freq} \
                   ${dim_hidden} \
                   ${dim_context} \
                   ${ctx_wt} \
                   ${stickiness} \
                   ${concentration} \
                   ${lik_softmax_beta} \
                   ${try_reset_h} \
                   ${use_shortcut} \
                   ${gen_grad} \
                   ${pe_tracker_size} \
                   ${match_tracker_size} \
                   ${n_pe_std} \
                   ${exp_name}

            done
          done
        done
      done
    done
  done
done
