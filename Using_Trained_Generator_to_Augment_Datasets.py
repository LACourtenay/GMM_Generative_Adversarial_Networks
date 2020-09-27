# set directory
import os
from os import listdir
from os.path import isfile, join
import glob
directory = "C:/Users/" # complete directory
os.chdir(directory)

# load libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Input, Concatenate
from tensorflow.keras.layers import LeakyReLU, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import math
from numpy.random import randint, rand, randn, random, choice
from numpy import ones, zeros, vstack
import time
import matplotlib
from matplotlib import pyplot as plt

# Print package versions and ensure they have loaded correctly
print("Versions:")
print("Tensorflow %s" % tf.__version__)
print("Keras %s" % keras.__version__)
print("Numpy %s" % np.__version__)
print("Matplotlib %s" % matplotlib.__version__)
print("Pandas %s" % pd.__version__)

# Set random number generator
seed_value = 100
np.random.seed(seed_value)

# If you want to alter the number of rows shown in pandas:
# pd.set_option("display.max_rows", 200)

# Matplotlib style = ggplot
plt.style.use("ggplot")

# Recall generate latent points function

latent_dim = 50 # remember to set the latent dimensions to the same value used during training

def generate_latent_points(n, latent_dim):
    x_input = rand(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

# Load data

dataset = pd.read_csv("scaled_data.txt", header = None)
X = pd.DataFrame.to_numpy(dataset.loc[:,0:dataset.shape[1]], dtype = "float")

# Plot data

plt.scatter(X[:, 0], X[:, 1], color = "blue")

# For a group of generators that the user wishes to test statistically, they can:
# Load all generators in directory and augment data

model_list = glob.glob("*.h5")
new_data = pd.DataFrame()
for i in range(len(model_list)):
    model = load_model(model_list[i])
    latent_points = generate_latent_points(60)
    data = model.predict(latent_points)
    data_df = pd.DataFrame(data)
    data_df.insert(0, "Label", model_list[i])
    new_data = new_data.append(data_df)
new_data.to_csv("Augmented_data.csv", sep = ",", index = False)

# If the user already knows which generator is the optimal, then a single generator
# can be loaded and used to augment the dataset

generator = load_model("generator.h5") # load the single generator model

new_individuals = 100 #set the number of individuals or times the data will be augmented

# produce new data
latent_points = generate_latent_points(100, latent_dim = 75)
new_data = generator.predict(latent_points)

# Create database with original and synthetic data
index_names = []
labels_1 = []
labels_2 = []
for i in range(X.shape[1]):
    index_names.append("PC" + str(i+1))
for i in range(data.shape[0]):
    labels_1.append("Synthetic")
for i in range(X.shape[0]):
    labels_2.append("Real")
synth_data = pd.DataFrame(data, columns = [index_names])
synth_data["label"] = labels_1
original_data = pd.DataFrame(X, columns = [index_names])
original_data["label"] = labels_2
final_data = pd.concat([synth_data, original_data], ignore_index = True)

# plot original and synthetic data - Red is synthetic data, black is real

plt.scatter(synth_data.iloc[:, 0], synth_data.iloc[:, 1], color = "red")
plt.scatter(X[:,0],X[:,1],color = "black")
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.show()

# Save final augmented dataset

final_data.to_csv("Final_augmented_data.csv", index = False, header = True)