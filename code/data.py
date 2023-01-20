import torch
from torch.utils.data import Dataset
import pandas as pd


class ClassificationDataset(Dataset):
    """
    A Dataset for classification
    """

    def __init__(self, path_to_dataset, label, tokenizer, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param label: the quality dimension to be predicted
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.labels = self.dataset[label].values
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).type(torch.LongTensor)
        # labels = labels.type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.labels)




class InferenceDataset(Dataset):
    """
       A Dataset for inference
       """

    def __init__(self, path_to_dataset, tokenizer, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # labels = labels.type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.dataset)


class RegressionDataset(Dataset):
    """
        A Europolis Dataset. Takes a pandas dataframe and loads the features, labels and encodings given the tokenizer
        """

    def __init__(self, path_to_dataset, label, tokenizer, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param label: the quality dimension to be predicted
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.labels = self.dataset[label].values
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # convert to float32 for regression
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)
