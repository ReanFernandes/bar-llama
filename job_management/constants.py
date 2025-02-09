from pathlib import Path

BASE_DIR = Path("/home/fr/fr_fr/fr_rf1031/bar-llama")
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