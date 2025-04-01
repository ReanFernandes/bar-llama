import json
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import logging
# note to self:
# args list for this file:
# data_dir
# to add the following functionalities:
# num_questions : number of questions to be included in the dataset [DONE]
# domains : list of domains to be included in the dataset [DONE]
# batch_size [DONE]
# random question selection [DONE]


class QuestionDataset(Dataset):
    """
    Represents a dataset of questions loaded from multiple JSON files,
    supporting domain-based filtering and random/deterministic sampling.
    """

    def __init__(self, config):
        """ ""
        Initializes the dataset.

        Args:

        """
        self.config = config
        self.config["num_questions"] = None if self.config["num_questions"] == "None" else self.config["num_questions"]
        self.data_points = []
        self.domain_indices = {}  # this dict contains the indices for all questions belonging to a specific domain in the dataset
        self.dataset_path = self.config["dataset_path"]
        logging.info(f"Loading dataset from {self.dataset_path}")
        # load entire dataset
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            all_data = json.load(file)
        logging.info("Dataset loaded successfully, indexing questions...")
        self._index_all_questions(all_data)
        logging.info(f"Indexing complete, sampling of questions set to {self.config['randomise_questions']}")
        self._select_questions(all_data)
        self.len = self.__len__()
        logging.info(f"Questions selected, dataset contains {self.len} questions with {self.len/(len(self.domain_indices))} questions per domain")
        

    def _index_all_questions(self, all_data):
        """
        Gonna use this function to find the start and end indexes of each unique domain in the dataset
        """
        df = pd.DataFrame(all_data)
        # find the unique domain names in the domain field
        unique_domains = df["domain"].unique()
        # create a dictionary to store the start and end indexes of each domain
        domain_indices = {}
        for domain in unique_domains:
            # get the indexes of the domain
            indices = df.index[df["domain"] == domain].tolist()
            # store the indexes in the dictionary
            domain_indices[domain] = indices
        self.domain_indices = domain_indices

    def _select_questions(self, all_data):
        """
        Selects a subset of questions from the dataset based on the specified configuration.
        Handles cases where a domain has fewer questions than the requested number.
        """
        if "all" in self.config["domains"]:
            if self.config["num_questions"]:
                if self.config["randomise_questions"]:
                    # Random selection of questions within each domain
                    for _, indices in self.domain_indices.items():
                        num_to_select = min(self.config["num_questions"], len(indices))  # Handle small domains
                        selected_indices = np.random.choice(indices, num_to_select, replace=False)
                        self.data_points.extend([all_data[i] for i in selected_indices])
                else:
                    # Select questions sequentially from the start of each domain
                    for _, indices in self.domain_indices.items():
                        self.data_points.extend([all_data[i] for i in indices[: self.config["num_questions"]]])
            else:
                # Select all questions if no limit is specified
                self.data_points.extend(all_data[i] for indices in self.domain_indices.values() for i in indices)
        else:
            # Filter by specific domains
            if self.config["num_questions"]:
                if self.config["randomise_questions"]:
                    for domain in self.config["domains"]:
                        indices = self.domain_indices[domain]
                        num_to_select = min(self.config["num_questions"], len(indices))  # Handle small domains
                        selected_indices = np.random.choice(indices, num_to_select, replace=False)
                        self.data_points.extend([all_data[i] for i in selected_indices])
                else:
                    for domain in self.config["domains"]:
                        indices = self.domain_indices[domain]
                        self.data_points.extend([all_data[i] for i in indices[: self.config["num_questions"]]])
            else:
                for domain in self.config["domains"]:
                    indices = self.domain_indices[domain]
                    self.data_points.extend([all_data[i] for i in indices])

    def __len__(self):
        """
        Returns the total number of questions in the dataset.
        """

        return len(self.data_points)

    def __getitem__(self, index):
        """
        Returns a single question data point at the specified index.

        Args:
            index (int): The index of the question to retrieve.
        """
        return self.data_points[index]

    def custom_collate_fn(self, batch):
        """
        Custom collate function to be used with the DataLoader.
        """
        return batch[0]
