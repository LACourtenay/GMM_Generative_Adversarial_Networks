# libraries and directory

import pandas as pd
import numpy as np
import os

directory = "C:/Users/" # complete directory
os.chdir(directory)

# load data

dataset = pd.read_csv("data.txt", header = None)

# scale data so that values fall between -1 and 1

X = pd.DataFrame.to_numpy(dataset.loc[:,0:dataset.shape[1]], dtype = "float")
X_std = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
X = X_std * (1 - -1) + -1

# save scaled data

np.savetxt("scaled_data.txt", X, delimiter = ",")