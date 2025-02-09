# import subprocess
# import os
# from pathlib import Path
# import logging
# from typing import Dict, Optional
# from .constants import JobState, BASE_DIR, LOGS_DIR, SCRIPTS_DIR
# from .database import DatabaseManager

# logger = logging.getLogger(__name__)

# class TrainEvalJobManager:
#     def __init__(self, base_dir: Path, queue_limit: int = 96, 
#                  buffer_size: int = 16):
#         self.base_dir = Path(base_dir)
#         self.queue_limit = queue_limit
#         self.buffer_size = buffer_size
#         self.target_queue_size = queue_limit - buffer_size
        
#         self.db = DatabaseManager(self.base_dir / "database" / "jobs.db")
        
#         for directory in [LOGS_DIR, SCRIPTS_DIR]:
#             directory.mkdir(parents=True, exist_ok=True)

#     def get_queue_status(self) -> Optional[Dict[str, int]]:
#         try:
#             result = subprocess.run(
#                 ['squeue', '-u', os.getenv('USER'), '-h', '-o', '%T'],
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )
#             states = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
#             return {
#                 'running': states.count('RUNNING'),
#                 'queued': len(states),
#                 'available': self.target_queue_size - len(states)
#             }
#         except subprocess.CalledProcessError as e:
#             logger.error(f"Error getting queue status: {e}")
#             return None

#     def _create_job_script(self, job_id: int, config: str) -> Path:
#         script_path = SCRIPTS_DIR / f"job_{job_id}.sh"
        
#         script_content = f"""#!/bin/bash
# #SBATCH --job-name=ml-{job_id}
# #SBATCH --output={LOGS_DIR}/job_{job_id}.out
# #SBATCH --error={LOGS_DIR}/job_{job_id}.err
# #SBATCH --partition=gpu_8
# #SBATCH --gres=gpu:1
# #SBATCH --mem=36G
# #SBATCH --time=47:45:00

# echo "Starting job {job_id} at $(date)"
# echo "Configuration: {config}"

# # Training phase
# python3 {self.base_dir}/scripts/update_status.py {job_id} training
# python3 {self.base_dir}/train.py {config}
# TRAIN_EXIT=$?

# if [ $TRAIN_EXIT -eq 0 ]; then
#     python3 {self.base_dir}/scripts/update_status.py {job_id} trained $TRAIN_EXIT
    
#     # Get and run evaluation configs
#     python3 {self.base_dir}/scripts/run_evaluations.py {job_id}
# else
#     python3 {self.base_dir}/scripts/update_status.py {job_id} failed $TRAIN_EXIT
# fi

# python3 {self.base_dir}/scripts/check_queue.py

# echo "Job {job_id} completed at $(date)"
# """
#         with open(script_path, 'w') as f:
#             f.write(script_content)
        
#         return script_path


#     # def submit_jobs(self, available_slots: int) -> int:
#     #     """Submit jobs to SLURM queue"""
#     #     pending_jobs = self.db.get_pending_jobs()[:available_slots]
#     #     submitted = 0
        
#     #     for job_id in pending_jobs:
#     #         config = self.db.get_train_config_string(job_id)
#     #         script_path = self._create_job_script(job_id, config)
            
#     #         try:
#     #             result = subprocess.run(['sbatch', str(script_path)], 
#     #                                 capture_output=True, 
#     #                                 text=True, 
#     #                                 check=True)
#     #             submitted += 1
#     #         except subprocess.CalledProcessError as e:
#     #             logger.error(f"Failed to submit job {job_id}: {e}")
                
#     #     return submitted

import subprocess
import os
from pathlib import Path
import logging
from typing import Dict, Optional, List
from .constants import JobState, BASE_DIR, LOGS_DIR, SCRIPTS_DIR
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class TrainEvalJobManager:
    def __init__(self, base_dir: Path, queue_limit: int = 96, 
                 buffer_size: int = 16):
        self.base_dir = Path(base_dir)
        self.queue_limit = queue_limit
        self.buffer_size = buffer_size
        self.target_queue_size = queue_limit - buffer_size
        
        self.db = DatabaseManager(self.base_dir / "database" / "jobs.db")
        
        for directory in [LOGS_DIR, SCRIPTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_queue_status(self) -> Optional[Dict[str, int]]:
        try:
            result = subprocess.run(
                ['squeue', '-u', os.getenv('USER'), '-h', '-o', '%T'],
                capture_output=True,
                text=True,
                check=True
            )
            states = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            return {
                'running': states.count('RUNNING'),
                'queued': len(states),
                'available': self.target_queue_size - len(states)
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting queue status: {e}")
            return None

    def _create_job_script(self, job_id: int, config: str) -> Path:
        script_path = SCRIPTS_DIR / f"job_{job_id}.sh"
        
        # Get eval configs upfront to include in the script
        eval_configs = self.db.get_eval_configs(job_id)
    
        script_content = f"""#!/bin/bash
#SBATCH --job-name=ml-{job_id}
#SBATCH --output={LOGS_DIR}/job_{job_id}.out
#SBATCH --error={LOGS_DIR}/job_{job_id}.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=47:45:00
echo "Starting job {job_id} at $(date)"
echo "Configuration: {config}"

# Training phase
python3 {self.base_dir}/scripts/update_status.py {job_id} training
python3 {self.base_dir}/train.py {config}
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    python3 {self.base_dir}/scripts/update_status.py {job_id} trained $TRAIN_EXIT
    
    # Run evaluations
    echo "Starting evaluations for job {job_id}"
    """
    
    # Add each eval config as a separate command
        for eval_config in eval_configs:
            script_content += f"""
echo "Running evaluation with config: {eval_config}"
# Update eval status to running
python3 {self.base_dir}/scripts/update_eval_status.py {job_id} "{eval_config}" "running"

python3 {self.base_dir}/eval.py {eval_config}
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 {self.base_dir}/scripts/update_eval_status.py {job_id} "{eval_config}" "completed" $EVAL_EXIT
else
    python3 {self.base_dir}/scripts/update_eval_status.py {job_id} "{eval_config}" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: {eval_config}"
fi
"""
    
        script_content += f"""
else
    python3 {self.base_dir}/scripts/update_status.py {job_id} failed $TRAIN_EXIT
fi

python3 {self.base_dir}/scripts/check_queue.py

echo "Job {job_id} completed at $(date)"
"""
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def submit_jobs(self, available_slots: int) -> int:
        """Submit jobs to SLURM queue"""
        pending_jobs = self.db.get_pending_jobs()[:available_slots]
        submitted = 0
        
        for job_id in pending_jobs:
            config = self.db.get_train_config_string(job_id)
            if not config:
                logger.error(f"Could not get config for job {job_id}")
                continue
                
            script_path = self._create_job_script(job_id, config)
            
            try:
                result = subprocess.run(
                    ['sbatch', str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Extract SLURM job ID and update database
                slurm_id = result.stdout.strip().split()[-1]
                self.db.update_job_status(job_id, JobState.QUEUED, slurm_id=slurm_id)
                submitted += 1
                logger.info(f"Submitted job {job_id} (SLURM ID: {slurm_id})")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit job {job_id}: {e}")
                
        return submitted

    def update_job_status(self, job_id: int, status: str, exit_code: Optional[int] = None):
        """Update status of a job"""
        self.db.update_job_status(job_id, status, exit_code=exit_code)
