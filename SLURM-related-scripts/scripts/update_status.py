
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from job_management.manager import TrainEvalJobManager
from job_management.constants import BASE_DIR

def main():
    """Update job status from SLURM script"""
    print("Starting update_status.py")
    print(f"Received arguments: {sys.argv}")
    
    if len(sys.argv) < 3:
        print("Usage: update_status.py <job_id> <status_type> [exit_code]")
        sys.exit(1)
        
    job_id = int(sys.argv[1])
    status_type = sys.argv[2]
    exit_code = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"Updating job {job_id} to status {status_type} with exit code {exit_code}")
    
    manager = TrainEvalJobManager(BASE_DIR)
    manager.db.update_job_status(job_id, status_type, exit_code=exit_code)
    
    print(f"Status update completed for job {job_id}")

if __name__ == "__main__":
    main()