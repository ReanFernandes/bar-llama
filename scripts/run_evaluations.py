import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_management import TrainEvalJobManager, BASE_DIR

def main():
    if len(sys.argv) != 2:
        print("Usage: run_evaluations.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    manager = TrainEvalJobManager(BASE_DIR)
    manager.run_evaluations(job_id)

if __name__ == "__main__":
    main()