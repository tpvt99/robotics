#!/usr/bin/env bash
# YOU SHOULD CHANGE HalfCheetah to Swimmer, Hopper to test on other two envs
# YOU SHOULD CHANGE exp_name for each subproblem. Otherwise, your result will be wiped.
export PYTHONPATH="/home/tpvt96/ai_course/robotics/cs287hw5/hw5_release_v2/ppo_tf2:$PYTHONPATH"
for i in `seq 0 0`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name HalfCheetah \
                               --exp_name mbppo_new \
                               --exp_num $i \
                               --ensemble 1
done