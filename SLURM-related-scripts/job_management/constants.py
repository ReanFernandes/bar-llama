from pathlib import Path

BASE_DIR = Path("/pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama")
LOGS_DIR = BASE_DIR / "logs"
DATABASE_DIR = BASE_DIR / "database"
SCRIPTS_DIR = BASE_DIR / "job_scripts"

class JobState:
    PENDING = 'pending'
    QUEUED = 'queued'
    TRAINING = 'training'
    TRAINED = 'trained'
    EVALUATING = 'evaluating'
    COMPLETED = 'completed'
    FAILED = 'failed'