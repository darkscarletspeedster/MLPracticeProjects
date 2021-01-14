import numpy as np
import pandas as pd
import matplotlib as plt

#importing data file
dataset = pd.read_csv('Data.csv')

#spliting dependent and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values