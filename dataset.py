import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import train_test_split
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import time
import os
import pandas as pd

class MatchesDataset(Dataset):
    def __init__(self, training=True):
        self.training = training
        matches = pd.read_csv("matches/matches.csv", index_col = False)
        y_data = matches['radiant_win']
        y_data = y_data.values
        x_data = matches.drop(['Unnamed: 0','duration', 'loss', 'match_id', 'match_seq_num', 'patch','radiant_win','region','skill','start_time','throw','radiant_team','dire_team'],1)
        # x_data = (x_data-x_data.mean())/x_data.std()
        assert(not x_data.isnull().values.any())
        self.x_data, self.x_test, self.y_data, self.y_test = train_test_split(x_data, y_data,
                                                        test_size = int(0.1*x_data.shape[0]),
                                                        random_state = 2,
                                                        stratify = y_data)
    def __len__(self):
        if (self.training):
            return self.x_data.shape[0]
        else:
            return self.x_test.shape[0]

    def __getitem__(self, index):
        if (self.training):
            return torch.from_numpy(self.x_data.iloc[index].values), torch.from_numpy(np.asarray(self.y_data[index].astype(int)))
        else:
            return torch.from_numpy(self.x_test.iloc[index].values), torch.from_numpy(np.asarray(self.y_test[index].astype(int)))
            

# dataset =  MatchesDataSet(training=True)
# print(dataset.__getitem__(100))