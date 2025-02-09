import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If database exists but is corrupted, remove it
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path)
                conn.close()
            except sqlite3.Error:
                print(f"Existing database corrupted, removing {self.db_path}")
                self.db_path.unlink()
        
        self._setup_database()
    def _setup_database(self):
        try:    
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    base_config TEXT,
                    dataset TEXT,
                    train_config_string TEXT,
                    status TEXT DEFAULT 'pending',
                    slurm_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    train_exit_code INTEGER,
                    current_eval_dataset TEXT,
                    error_message TEXT,
                    UNIQUE(base_config, dataset)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS eval_configs (
                    job_id INTEGER,
                    base_config TEXT,
                    eval_dataset TEXT,
                    seed TEXT,
                    generation TEXT,
                    quantisation TEXT,
                    training_status TEXT,
                    config_string TEXT,
                    status TEXT DEFAULT 'pending',
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    exit_code INTEGER,
                    error_message TEXT,
                    FOREIGN KEY(job_id) REFERENCES jobs(id),
                    UNIQUE(job_id, base_config, eval_dataset, seed, generation, quantisation, training_status)
                )
            ''')
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Error setting up database: {e}")
            raise
        finally:
            conn.close()
    def add_job_pair(self, config_pair: Dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO jobs (base_config, dataset, train_config_string)
                VALUES (?,?,?)
            ''', (
                config_pair['identifiers']['base_config'],
                config_pair['identifiers']['dataset'],
                config_pair['train_config_string']

            ))
            job_id = c.lastrowid

            for eval_config in config_pair['eval_configs']: 
                c.execute('''
                    INSERT INTO eval_configs(
                    job_id, base_config, eval_dataset, seed, generation,
                    quantisation, training_status, config_string ) VALUES(?,?,?,?,?,?,?,?)
                    ''', (
                        job_id,
                        config_pair['identifiers']['base_config'],  
                        eval_config['identifiers']['eval_dataset'],
                        eval_config['identifiers']['seed'],
                        eval_config['identifiers']['generation'],
                        eval_config['identifiers']['quantisation'],
                        eval_config['identifiers']['training_status'],
                        eval_config['config_string']
                    ))  
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error adding entry to database: {e}")
            raise
        finally:
            conn.close()
    
#     def get_job_config(self, job_id: int) -> str:
#         """Get train_config_string for a job"""
#         return self.db.get_train_config_string(job_id)  # This method needs to exist in DatabaseManager


# import sqlite3
# from pathlib import Path
# import logging
# from typing import Dict, List, Optional
# from datetime import datetime

# logger = logging.getLogger(__name__)

# class DatabaseManager:
#     def __init__(self, db_path: Path):
#         self.db_path = Path(db_path)
#         self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
#         if self.db_path.exists():
#             try:
#                 conn = sqlite3.connect(self.db_path)
#                 conn.close()
#             except sqlite3.Error:
#                 print(f"Existing database corrupted, removing {self.db_path}")
#                 self.db_path.unlink()
        
#         self._setup_database()

#     def _setup_database(self):
#         conn = sqlite3.connect(self.db_path)
#         c = conn.cursor()
        
#         try:
#             c.execute('''
#                 CREATE TABLE IF NOT EXISTS jobs (
#                     id INTEGER PRIMARY KEY,
#                     base_config TEXT,
#                     dataset TEXT,
#                     train_config_string TEXT,
#                     status TEXT DEFAULT 'pending',
#                     slurm_id TEXT,
#                     start_time TIMESTAMP,
#                     end_time TIMESTAMP,
#                     train_exit_code INTEGER,
#                     current_eval_dataset TEXT,
#                     error_message TEXT,
#                     UNIQUE(base_config, dataset)
#                 )
#             ''')
            
#             c.execute('''
#                 CREATE TABLE IF NOT EXISTS eval_configs (
#                     job_id INTEGER,
#                     base_config TEXT,
#                     eval_dataset TEXT,
#                     seed TEXT,
#                     generation TEXT,
#                     quantisation TEXT,
#                     training_status TEXT,
#                     config_string TEXT,
#                     status TEXT DEFAULT 'pending',
#                     exit_code INTEGER,
#                     FOREIGN KEY(job_id) REFERENCES jobs(id),
#                     UNIQUE(job_id, base_config, eval_dataset, seed, generation, quantisation, training_status)
#                 )
#             ''')
            
#             conn.commit()
#         finally:
#             conn.close()

#     def add_job_pair(self, config_pair: Dict):
#         conn = sqlite3.connect(self.db_path)
#         c = conn.cursor()
        
#         try:
#             c.execute('''
#                 INSERT INTO jobs (base_config, dataset, train_config_string)
#                 VALUES (?, ?, ?)
#             ''', (
#                 config_pair['identifiers']['base_config'],
#                 config_pair['identifiers']['dataset'],
#                 config_pair['train_config_string']
#             ))
#             job_id = c.lastrowid

#             for eval_config in config_pair['eval_configs']:
#                 c.execute('''
#                     INSERT INTO eval_configs (
#                         job_id, base_config, eval_dataset, seed, generation,
#                         quantisation, training_status, config_string
#                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#                 ''', (
#                     job_id,
#                     config_pair['identifiers']['base_config'],
#                     eval_config['identifiers']['eval_dataset'],
#                     eval_config['identifiers']['seed'],
#                     eval_config['identifiers']['generation'],
#                     eval_config['identifiers']['quantisation'],
#                     eval_config['identifiers']['training_status'],
#                     eval_config['config_string']
#                 ))
            
#             conn.commit()
#             return job_id
#         finally:
#             conn.close()

    def get_pending_jobs(self) -> List[int]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('SELECT id FROM jobs WHERE status = "pending" ORDER BY id')
            return [row[0] for row in c.fetchall()]
        finally:
            conn.close()

    def get_train_config_string(self, job_id: int) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('SELECT train_config_string FROM jobs WHERE id = ?', (job_id,))
            result = c.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def update_job_status(self, job_id: int, status: str, slurm_id: Optional[str] = None, 
                         exit_code: Optional[int] = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            updates = ['status = ?']
            params = [status]
            
            if slurm_id is not None:
                updates.append('slurm_id = ?')
                params.append(slurm_id)
            
            if exit_code is not None:
                updates.append('train_exit_code = ?')
                params.append(exit_code)
            
            if status in ['training', 'queued']:
                updates.append('start_time = CURRENT_TIMESTAMP')
            elif status in ['trained', 'failed']:
                updates.append('end_time = CURRENT_TIMESTAMP')
            
            query = f'''
                UPDATE jobs 
                SET {', '.join(updates)}
                WHERE id = ?
            '''
            params.append(job_id)
            
            c.execute(query, params)
            conn.commit()
        finally:
            conn.close()
