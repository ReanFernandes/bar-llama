import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import shutil
from job_management.manager import TrainEvalJobManager
from job_management.constants import JobState, BASE_DIR

def test_status_updates():
    # Set up test paths
    test_base_dir = Path("test_dir")
    test_db_dir = test_base_dir / "database"
    test_db_path = test_db_dir / "jobs.db"
    
    # Clean up any previous test files
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)
    
    # Create test directory structure
    test_db_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created test directory at: {test_base_dir}")
    
    # Create manager with test database
    manager = TrainEvalJobManager(test_base_dir)
    
    # Insert a job directly into database like it would exist in real system
    conn = sqlite3.connect(test_db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO jobs (id, base_config, dataset, model, train_config_string, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (30, 'test_config', 'test_dataset', 'test_model', 'model=test', 'pending'))
    conn.commit()
    
    print("\nTest 1: Current method (using manager wrapper)")
    print("Running: manager.update_job_status(30, JobState.TRAINED, 0)")
    manager.update_job_status(30, JobState.TRAINED, 0)
    
    c.execute("SELECT status FROM jobs WHERE id = 30")
    status1 = c.fetchone()[0]
    print(f"Status after manager update: {status1}")
    
    # Reset status for next test
    c.execute("UPDATE jobs SET status = 'pending' WHERE id = 30")
    conn.commit()
    
    print("\nTest 2: Direct database update")
    print("Running: manager.db.update_job_status(30, JobState.TRAINED, exit_code=0)")
    manager.db.update_job_status(30, JobState.TRAINED, exit_code=0)
    
    c.execute("SELECT status FROM jobs WHERE id = 30")
    status2 = c.fetchone()[0]
    print(f"Status after direct db update: {status2}")
    
    # Clean up
    conn.close()
    shutil.rmtree(test_base_dir)
    print("\nTest cleanup completed")

if __name__ == "__main__":
    test_status_updates()