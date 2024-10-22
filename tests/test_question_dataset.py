"""
This module contains unit tests for the QuestionDataset class.

The tests cover various aspects of the QuestionDataset functionality, including:
- Loading all domains
- Filtering specific domains
- Limiting the number of questions
- Randomizing question selection

These tests ensure that the QuestionDataset class correctly loads and filters
data based on the question_numbered configuration options.
"""


import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from utils.dataloader import QuestionDataset

class TestQuestionDataset(unittest.TestCase):
    """Test cases for the QuestionDataset class."""

    @classmethod
    def setUpClass(cls):
        # Set up the path to your existing dataset
        # cls.dataset_path = "/mnt/b1aa7336-e92e-43c1-9d8b-d627549a7f7d/bar-llama/dataset/seed_dataset/high_temp_structured_expl_dataset.json" # this is on my local pc
        cls.dataset_path = "/work/dlclarge1/fernandr-thesis_workspace/bar-llama/dataset/seed_dataset/high_temp_structured_expl_dataset.json"    # this is on the server

    def test_all_domains(self):
        config = {
            "dataset_path": self.dataset_path,
            "domains": ["all"],
            "num_questions": None,
            "randomise_questions": False
        }
        dataset = QuestionDataset(config)
        self.assertGreater(len(dataset), 0)

    def test_specific_domains(self):
        # Assuming your dataset has at least two domains, replace with actual domain names
        test_domains = ["Criminal_law", "Evidence"]
        config = {
            "dataset_path": self.dataset_path,
            "domains": test_domains,
            "num_questions": None,
            "randomise_questions": False
        }
        dataset = QuestionDataset(config)
        self.assertGreater(len(dataset), 0)
        for item in dataset:
            self.assertIn(item['domain'], test_domains)

    def test_num_questions_limit(self):
        config = {
            "dataset_path": self.dataset_path,
            "domains": ["all"],
            "num_questions": 5,
            "randomise_questions": False
        }
        dataset = QuestionDataset(config)
        unique_domains = set(item['domain'] for item in dataset)
        self.assertLessEqual(len(dataset), len(unique_domains) * 5)

    def test_randomise_questions(self):
        config = {
            "dataset_path": self.dataset_path,
            "domains": ["all"],
            "num_questions": 10,
            "randomise_questions": True
        }
        dataset1 = QuestionDataset(config)
        dataset2 = QuestionDataset(config)
        # This test might occasionally fail due to randomness
        self.assertNotEqual([item['question_number'] for item in dataset1], [item['question_number'] for item in dataset2])

    def test_getitem(self):
        config = {
            "dataset_path": self.dataset_path,
            "domains": ["all"],
            "num_questions": None,
            "randomise_questions": False
        }
        dataset = QuestionDataset(config)
        item = dataset[0]
        self.assertIn("question_number", item)
        self.assertIn("question", item)
        self.assertIn("domain", item)

    def test_custom_collate_fn(self):
        config = {
            "dataset_path": self.dataset_path,
            "domains": ["all"],
            "num_questions": None,
            "randomise_questions": False
        }
        dataset = QuestionDataset(config)
        batch = [dataset[0], dataset[1]]
        collated = dataset.custom_collate_fn(batch)
        self.assertEqual(collated, dataset[0])

if __name__ == '__main__':
    unittest.main()