Once submit_jobs.py is run, this is the workflow

1) A database is initialised
2) selected configuration values are combined. the subsequent training configs and their associate evaluation configs are written to this database
3) The max limit (80 jobs in our case) of jobs is created in job_management/manager and the bash script for each job is modified 
4) Jobs are submitted. Once each job starts, it updates it status to the database (queued, running, pending, completed) for both the training and evaluation. 
5) In any one job, first the fine-tuning of the model is done. Then the untrained evaluations, followed by the fine-tuned model evaluations, after which the job is completed. 
6) Once the job is completed, it updated its status as completed, and triggers the manager to cheeck the queue status, and if below the job threshold, submit a new job. 

