import torch
import pandas as pd
from torch import nn
from d2l import torch as d2l

data = pd.read_csv("a.us.txt")
print(data.head())
length_data = len(data)     # rows that data has
split_ratio = 0.7           # %70 train + %30 validation
length_train = round(length_data * split_ratio)
length_validation = length_data - length_train
print("Data length :", length_data)
print("Train data length :", length_train)
print("Validation data lenth :", length_validation)