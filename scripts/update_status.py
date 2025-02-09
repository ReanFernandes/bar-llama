import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from job_management.manager import TrainEvalJobManager
from job_management.constants import BASE_DIR

def main():
    """Update job status from SLURM script"""
    if len(sys.argv) < 3:
        print("Usage: update_status.py <job_id> <status_type> [exit_code]")
        sys.exit(1)
        
    job_id = int(sys.argv[1])
    status_type = sys.argv[2]
    exit_code = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    manager = TrainEvalJobManager(BASE_DIR)
    manager.update_job_status(job_id, status_type, exit_code)
