import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


def load_and_preprocess_data(training_data_path, test_data_path): 
   train_data = pd.read_csv(training_data_path)
   test_data = pd.read_csv(test_data_path)

   # we combine train and test data for preprocessing
   # first column is ID, last column is SalePrice. Test data only has ID and features.
   all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

   # standardize the numeric features
   # you can read more details https://www.geeksforgeeks.org/what-is-standardization-in-machine-learning/
   # this trick is very common in machine learning but not in deep learning
   numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
   all_features[numeric_features] = all_features[numeric_features].apply(
      lambda x: (x - x.mean()) / (x.std()))
   all_features[numeric_features] = all_features[numeric_features].fillna(0) # I just use 0 for missing values, you can use other methods

   # one-hot encode categorical features
   all_features = pd.get_dummies(all_features, dummy_na=True)
   all_features.shape # (2919, 331)


   n_train = train_data.shape[0] # this a number of training examples, we will use it later

   # convert True/False to 1/0
   all_features.replace({False: 0, True: 1}, inplace=True)

   # convert to tensors
   train_features = torch.from_numpy(all_features[:n_train].values)
   test_features = torch.from_numpy(all_features[n_train:].values)
   train_labels = torch.from_numpy(train_data.SalePrice.values).view(-1, 1) # -1 means infer the size of the second dimension

   # convert to float
   train_features = train_features.float()
   test_features = test_features.float()
   train_labels = train_labels.float()

   return train_features, test_features, train_labels

