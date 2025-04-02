import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from job_management.manager import TrainEvalJobManager
from job_management.constants import BASE_DIR

logger = logging.getLogger(__name__)

def main():
    """Check queue and submit new jobs if needed"""
    manager = TrainEvalJobManager(BASE_DIR)
    
    status = manager.get_queue_status()
    if not status:
        logger.error("Could not get queue status")
        return
        
    if status['available'] > 0:
        submitted = manager.submit_jobs(status['available'])
        logger.info(f"Submitted {submitted} new jobs")
    
    logger.info(f"Current Status - Running: {status['running']}, "
                f"Total in queue: {status['queued']}, "
                f"Available slots: {status['available']}")

if __name__ == "__main__":
    main()