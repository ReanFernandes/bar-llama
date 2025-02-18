import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from job_management.resubmit_manager import ResubmissionManager
from job_management.constants import BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    manager = ResubmissionManager(BASE_DIR)
    submitted = manager.submit_failed_jobs()
    logger.info(f"Resubmitted {submitted} failed jobs to SLURM queue")

if __name__ == "__main__":
    main()