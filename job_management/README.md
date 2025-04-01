*Note to users* 
This job management module was meant to be cluster specific, and was mainly used for chaining jobs together to conform with the maximum number of possible jobs permitted on the cluster. As such, it is only meant to be used on a SLURM based cluster ( but only after changes to the bash commands to conform according to the users cluster, bascially everything in sbatch must be adapted)

The way this script works is as follows: 
1) Configuration parameters are selected in config_generator. Any config value thats not to be test should be commented out
2) script/submit_jobs.py will be run which does the submission
