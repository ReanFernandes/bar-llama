# Flexible Evaluation System

## Structure
```
flexible_eval/
├── configs/
│   ├── config.py       # Configuration settings
│   └── all_configs.json    # Generated configurations
├── scripts/
│   ├── job_generator.py    # Main script
│   └── job_script.sh       # Generated job script
├── logs/                   # Job output logs
└── README.md
```

## Usage

1. Modify configurations in `configs/config.py`:
   - Update component values
   - Adjust SLURM settings
   - Change environment setup

2. Generate and submit jobs:
```bash
# Default: 40 jobs per batch, start with batch 1
python3 scripts/job_generator.py

# Custom jobs per batch
python3 scripts/job_generator.py --jobs-per-batch 30

# Start from specific batch
python3 scripts/job_generator.py --start-batch 2
```

3. Monitor jobs:
```bash
squeue -u $USER
```

## Examples

1. Disable specific components:
```python
from configs.config import ExperimentConfig

config = ExperimentConfig()
config.disable_component_values('seeds', ['seed_3991'])
```

2. Enable only specific components:
```python
config.enable_only_component_values('generation', ['greedy', 'temp_025'])
```

3. Update SLURM settings:
```python
config.update_slurm_config({
    'mem': '48G',
    'time': '2:00:00',
    'max_concurrent_jobs': 8
})
```
