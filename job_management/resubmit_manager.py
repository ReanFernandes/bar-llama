import sys
import os
import logging
from pathlib import Path
import sqlite3
from typing import Dict, List, Set, Tuple
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FailedJob:
    base_config: str
    dataset: str
    model: str
    train_config_string: str
    needs_training: bool
    eval_configs: List[Dict]

class ResubmissionManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "database" / "jobs.db"
        self.scripts_dir = self.base_dir / "scripts"
        self.logs_dir = self.base_dir / "logs"
        
        # Ensure directories exist
        for directory in [self.scripts_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_failed_jobs(self) -> List[FailedJob]:
        """Retrieve failed jobs and their associated failed eval configs."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        failed_jobs = []

        try:
            # Get jobs where either training failed or has failed eval configs
            c.execute('''
                SELECT id, base_config, dataset, model, train_config_string, status
                FROM jobs 
                WHERE status = 'failed' OR status = 'trained'
            ''')
            jobs = c.fetchall()

            for job_id, base_config, dataset, model, train_config_string, status in jobs:
                # Get failed eval configs for this job
                c.execute('''
                    SELECT config_string
                    FROM eval_configs 
                    WHERE job_id = ? AND status = 'failed'
                ''', (job_id,))
                
                failed_evals = [{"config_string": row[0]} for row in c.fetchall()]
                
                if status == 'failed' or failed_evals:  # Only include if training failed or has failed evals
                    failed_jobs.append(FailedJob(
                        base_config=base_config,
                        dataset=dataset,
                        model=model,
                        train_config_string=train_config_string,
                        needs_training=(status == 'failed'),  # Flag if training needs to be rerun
                        eval_configs=failed_evals  # Only include failed eval configs
                    ))

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

        return failed_jobs

    def _create_job_script(self, job: FailedJob, job_id: int) -> Path:
        """Create a SLURM job script for the failed job."""
        script_path = self.scripts_dir / f"resubmit_job_{job_id}.sh"
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=ml-{job_id}
#SBATCH --output={self.logs_dir}/job_{job_id}.out
#SBATCH --error={self.logs_dir}/job_{job_id}.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=47:45:00
echo "Starting resubmitted job {job_id} at $(date)"

# Set environment variables needed for both training and evaluation
export HUGGINGFACE_TOKEN="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"
export WANDB_PROJECT=Final_runs_paper

"""
        # Only include training if it failed in the original run
        if job.needs_training:
            script_content += f"""
echo "Configuration: {job.train_config_string}"

# Training phase
python3 {self.base_dir}/scripts/update_status.py {job_id} training
python3 {self.base_dir}/train.py {job.train_config_string}
TRAIN_EXIT=$?

python3 {self.base_dir}/scripts/update_status.py {job_id} trained $TRAIN_EXIT
"""

        # Run only the failed evaluations
        script_content += f"""
# Run failed evaluations
echo "Starting failed evaluations for job {job_id}"
"""
        
        # Add each failed eval config
        for eval_config in job.eval_configs:
            config_string = eval_config['config_string']
            script_content += f"""
echo "Running evaluation with config: {config_string}"
# Update eval status to running
python3 {self.base_dir}/scripts/update_eval_status.py {job_id} "{config_string}" "running"

python3 {self.base_dir}/eval.py {config_string}
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 {self.base_dir}/scripts/update_eval_status.py {job_id} "{config_string}" "completed" $EVAL_EXIT
else
    python3 {self.base_dir}/scripts/update_eval_status.py {job_id} "{config_string}" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: {config_string}"
fi
"""

        script_content += f"""
python3 {self.base_dir}/scripts/check_queue.py

echo "Job {job_id} completed at $(date)"
"""

        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path

    def submit_failed_jobs(self) -> int:
        """Submit all failed jobs to SLURM queue."""
        failed_jobs = self.get_failed_jobs()
        submitted = 0

        logger.info(f"Found {len(failed_jobs)} failed jobs to resubmit")
        
        for job_id, job in enumerate(failed_jobs, start=1):
            script_path = self._create_job_script(job, job_id)
            
            try:
                result = subprocess.run(
                    ['sbatch', str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                slurm_id = result.stdout.strip().split()[-1]
                logger.info(f"Submitted resubmission job {job_id} (SLURM ID: {slurm_id})")
                submitted += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit resubmission job {job_id}: {e}")

        return submitted