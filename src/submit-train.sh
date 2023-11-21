#!/bin/bash

ctx_wt=.5
lik_softmax_beta=0.33

# gen_grad=.5
# pe_tracker_size=32
# match_tracker_size=4
# n_pe_std=3
gen_grad=0.0
pe_tracker_size=0
match_tracker_size=0
n_pe_std=0.0
for subj_id in {0..11}
do
  for lr in 1e-3
  do
    for update_freq in 4
    do
      for dim_context in 128
      do
        for dim_hidden in 16
        do
          for stickiness in 4
          do
            for concentration in .75 1 1.5
            do
              for try_reset_h in 1 0
              do
                use_shortcut=0
                sbatch train.sh \
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
                     # use_shortcut=1
                     # for gen_grad in .15 .25 .35
                     #   do
                     #     for pe_tracker_size in 256
                     #     do
                     #       for match_tracker_size in 4
                     #       do
                     #         for n_pe_std in .5 1 2
                     #         do
                     #         sbatch train.sh \
                     #              ${subj_id} \
                     #              ${lr} \
                     #              ${update_freq} \
                     #              ${dim_hidden} \
                     #              ${dim_context} \
                     #              ${ctx_wt} \
                     #              ${stickiness} \
                     #              ${concentration} \
                     #              ${lik_softmax_beta} \
                     #              ${try_reset_h} \
                     #              ${use_shortcut} \
                     #              ${gen_grad} \
                     #              ${pe_tracker_size} \
                     #              ${match_tracker_size} \
                     #              ${n_pe_std} \
                     #              ${exp_name}
                     #        done
                     #      done
                     #    done
                     # done
              done
            done
          done
        done
      done
    done
  done
done
