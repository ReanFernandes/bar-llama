# At the top of submit_jobs.py
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_management.manager import TrainEvalJobManager
from job_management.config_generator import generate_train_eval_pairs
from job_management.constants import BASE_DIR

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    manager = TrainEvalJobManager(BASE_DIR)
    paired_configs = generate_train_eval_pairs()
    
    # First populate database with configs
    logger.info(f"Generated {len(paired_configs)} config pairs")
    for config_pair in paired_configs:
        manager.db.add_job_pair(config_pair)
    logger.info("Added all configs to database")
    
    # # Then submit jobs if queue has space
    status = manager.get_queue_status()
    if status and status['available'] > 0:
        submitted = manager.submit_jobs(status['available'])
        logger.info(f"Submitted {submitted} jobs to SLURM queue")
        logger.info(f"Queue status - Running: {status['running']}, "
                   f"Queued: {status['queued']}, "
                   f"Available: {status['available']}")

if __name__ == "__main__":
    main()