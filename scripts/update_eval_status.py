# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from job_management.manager import TrainEvalJobManager
# from job_management.constants import BASE_DIR

# def main():
#     if len(sys.argv) < 4:
#         print("Usage: update_eval_status.py <job_id> <config_string> <status> [exit_code] [error_message]")
#         sys.exit(1)
        
#     job_id = int(sys.argv[1])
#     config_string = sys.argv[2]
#     status = sys.argv[3]
#     exit_code = int(sys.argv[4]) if len(sys.argv) > 4 else None
#     error_message = sys.argv[5] if len(sys.argv) > 5 else None
    
#     manager = TrainEvalJobManager(BASE_DIR)
#     manager.db.update_eval_status(job_id, config_string, status, exit_code, error_message)

# if __name__ == "__main__":
#     main()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from job_management.manager import TrainEvalJobManager
from job_management.constants import BASE_DIR

def main():
    """Update evaluation status from SLURM script"""
    print("Starting update_eval_status.py")
    print(f"Received arguments: {sys.argv}")
    
    if len(sys.argv) < 4:
        print("Usage: update_eval_status.py <job_id> <config_string> <status> [exit_code] [error_message]")
        sys.exit(1)
        
    job_id = int(sys.argv[1])
    config_string = sys.argv[2]
    status = sys.argv[3]
    exit_code = int(sys.argv[4]) if len(sys.argv) > 4 else None
    error_message = sys.argv[5] if len(sys.argv) > 5 else None
    
    print(f"Updating eval for job {job_id} to status {status} with exit code {exit_code}")
    if error_message:
        print(f"Error message: {error_message}")
    
    manager = TrainEvalJobManager(BASE_DIR)
    manager.db.update_eval_status(job_id, config_string, status, exit_code, error_message)
    
    print(f"Eval status update completed for job {job_id}")

if __name__ == "__main__":
    main()