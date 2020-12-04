set PYTHONPATH=D:\GitHub\robotics\cs287hw5\hw5_release_v2\ppo_tf2
set env_name="Hopper"
set exp_name="ppo+entropy+gae"
set use_baseline=1
set use_ppo_obj=1
set use_clipper=1
set use_entropy=1
set use_gae=1


FOR  %%A IN (0,1,2) DO (
  ECHO %%A
  c:\Users\bptran\AppData\Local\Continuum\anaconda3\envs\rlcourse\python.exe ppo_run_sweep.py %* --env_name %env_name% ^
                            --exp_name %exp_name% ^
                            --exp_num %%A ^
                            --use_baseline %use_baseline% ^
                            --use_ppo_obj %use_ppo_obj% ^
                            --use_clipper %use_clipper% ^
                            --use_entropy %use_entropy% ^
                            --use_gae %use_gae%
)