import json
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
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
        self.data_points = []
        self.domain_indices = {}  # this dict contains the indices for all questions belonging to a specific domain in the dataset
        self.dataset_path = self.config["dataset_path"]

        # load entire dataset
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            all_data = json.load(file)

        self._index_all_questions(all_data)

        self._select_questions(all_data)
        self.len = self.__len__()

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
        Selects a subset of questions from the dataset based on the specified configuration. We first check if we are filtering by domain or selecting all the questions.
        after this is done we will check if there is a number of questions to be selected.
        if there is we will select the number of questions ( either randomly or the first n questions)
        if there is no number of questions to be selected we will select all the questions.
        """
        if "all" in self.config["domains"]:
            if self.config["num_questions"]:
                if  self.config["randomise_questions"] is True:  
                    # questions are randomly selected within the domain
                    # obtain the indexes of the questions to be selected.
                    # since we have the domain indexes, we will select self.config["num_questions"] from each domain
                    # and store in self.data_points
                    for _, indices in self.domain_indices.items():
                        selected_indices = np.random.choice(
                            indices, self.config["num_questions"], replace=False
                        )
                        self.data_points.extend([all_data[i] for i in selected_indices])
                else:  # No random choice of questions
                    # the questions are selected from the starting index of every domain
                    for _, indices in self.domain_indices.items():
                        selected_indices = indices[: self.config["num_questions"]]
                        self.data_points.extend([all_data[i] for i in selected_indices])
            else:  # all questions are selected
                self.data_points.extend(all_data[i] for indices in self.domain_indices.values() for i in indices)
        else:  # filtering by domain
            if self.config["num_questions"]:
                if self.config["randomise_questions"] is True:
                    for domain in self.config["domains"]:
                        indices = self.domain_indices[domain]
                        selected_indices = np.random.choice(
                            indices, self.config["num_questions"], replace=False
                        )
                        self.data_points.extend([all_data[i] for i in selected_indices])
                else:
                    for domain in self.config["domains"]:
                        indices = self.domain_indices[domain]
                        selected_indices = indices[: self.config["num_questions"]]
                        self.data_points.extend([all_data[i] for i in selected_indices])
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
