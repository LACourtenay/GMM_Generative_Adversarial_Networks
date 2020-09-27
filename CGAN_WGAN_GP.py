# set directory
import os
directory = "C:/Users/" # complete directory
os.chdir(directory)

# load libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Reshape, Input, Concatenate
from tensorflow.keras.layers import LeakyReLU, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.datasets.mnist import load_data
import pandas as pd
import numpy as np
import math
from numpy.random import randint, rand, randn, random, choice
from numpy import ones, zeros, vstack
import sklearn
from sklearn.preprocessing import LabelEncoder
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
print("SciKit Learn %s" % sklearn.__version__)

# Set random number generator
seed_value = 100
np.random.seed(seed_value)

# If you want to alter the number of rows shown in pandas:
# pd.set_option("display.max_rows", 200)

# Matplotlib style = ggplot
plt.style.use("ggplot")

# Define functions and models

# Define custom loss

def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# Define discriminator or "critic"

def define_critic_gp(dataset):
    
    init = RandomNormal(stddev = 0.1)
    
    feature_data = Input(shape = (dataset.shape[1],))
    label_data = Input(shape = (1,))
    
    label_embedding = Flatten()(Embedding(key.shape[0],
                                          math.ceil((1/4)*dataset.shape[1]))
                                (label_data))
    label_dense = Dense(dataset.shape[1]) (label_embedding)
    
    inputs = multiply([feature_data, label_dense])
    
    main_disc = Dense(math.ceil((1/2)*dataset.shape[1]),
                     kernel_initializer = init) (inputs)
    main_disc = BatchNormalization() (main_disc)
    main_disc = Activation("tanh") (main_disc)
    main_disc = Dense(math.ceil((1/4)*dataset.shape[1]),
                     kernel_initializer = init) (main_disc)
    main_disc = BatchNormalization() (main_disc)
    main_disc = Activation("tanh") (main_disc)
    main_disc = Dropout(0.4) (main_disc)
    disc_out = Dense(1, activation = "linear") (main_disc)
    
    discrim = Model([feature_data, label_data], disc_out)
    
    opt = RMSprop(lr = 0.00005)
    discrim.compile(loss = wasserstein_loss, optimizer = opt, metrics = ["accuracy"])
    return discrim

# Define generator

def define_generator(dataset, latent_dim, key):
    
    init = RandomNormal(stddev = 0.7)
    
    noise = Input(shape = (latent_dim,))
    label = Input(shape = (1,))
    
    label_embedding = Flatten()(Embedding(key.shape[0],
                                          math.ceil((1/4)*dataset.shape[1]))
                                (label))
    label_dense = Dense(latent_dim) (label_embedding)
    
    inputs = multiply([noise, label_dense])
    
    main_gen = Dense(math.ceil((1/4)*dataset.shape[1]),
                     kernel_initializer = init) (inputs)
    main_gen = BatchNormalization() (main_gen)
    main_gen = Activation("tanh") (main_gen)
    main_gen = Dense(math.ceil((1/2)*dataset.shape[1]),
                     kernel_initializer = init) (main_gen)
    main_gen = BatchNormalization() (main_gen)
    main_gen = Activation("tanh") (main_gen)
    main_gen = Dense((dataset.shape[1]+math.ceil((1/4)*dataset.shape[1])),
                      kernel_initializer = init) (main_gen)
    main_gen = BatchNormalization() (main_gen)
    main_gen = Activation("tanh") (main_gen)
    gen_out = Dense(dataset.shape[1], activation = "tanh") (main_gen)
    gen = Model([noise, label], gen_out)
    return gen

# Define GAN

def define_gan(generator, critic, latent_dim):
    noise = Input(shape = (latent_dim,))
    label = Input(shape = (1,))
    features = generator([noise, label])
    critic_valid = critic([features, label])
    critic.trainable = False
    gan_model = Model([noise, label], critic_valid)
    opt = RMSprop(lr = 0.000005)
    gan_model.compile(loss = wasserstein_loss, optimizer = opt, metrics = ["accuracy"])
    return gan_model

# Define functions for generation of real and fake samples

def generate_real_samples(dataset, n_samples, y_values):
    ix = randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    labels = y_values[ix]
    y = -ones((n_samples, 1))
    return [x, labels], y
def generate_fake_samples(generator, latent_dim, n, key):
    x_input, labels_input = generate_latent_points(latent_dim, n, key)
    x = generator.predict([x_input, labels_input])
    y = ones((n, 1))
    return [x, labels_input], y

# Define functions for generation of latent point inputs

def generate_latent_points(latent_dim, n, key):
    x_input = rand(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    labels = randint(0, key.shape[0], n)
    return [x_input, labels]

# Define function to summarize, save and checkpoint GAN performance

def summarize_performance(step, generator):
    h5_name = "generator_model_e%03d.h5" % (step + 1)
    generator.save(h5_name)

# Define function to plot training history

def plot_history(d1_hist, d2_hist, g_hist):
    plt.plot(d1_hist, label = "critic-real")
    plt.plot(d2_hist, label = "critic-fake")
    plt.plot(g_hist, label = "gen")
    plt.legend()
    plt.savefig("plot_line_plot_loss.png")

# Define final function for training

def wass_train(g_model, c_model, gan_model, dataset, latent_dim, key, y_values,
              n_epoch, n_batch, n_critic = 5, n_eval = 100):
    bat_per_epo = int(dataset.shape[0]/n_batch)
    n_steps = bat_per_epo * n_epoch
    half_batch = int(n_batch/2)
    c1_hist, c2_hist, g_hist = list(), list(), list()
    for i in range(n_steps):
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            [x_real, labels_real], y_real = generate_real_samples(dataset,
                                                                  half_batch,
                                                                  y_values)
            c_loss1, _ = c_model.train_on_batch([x_real, labels_real],
                                                y_real)
            c1_tmp.append(c_loss1)
            [x_fake, dif_labels], y_fake = generate_fake_samples(g_model,
                                                                 latent_dim,
                                                                 half_batch,
                                                                 key)
            c_loss2, _ = c_model.train_on_batch([x_fake, dif_labels],
                                                y_fake)
            c2_tmp.append(c_loss2)
        c1_hist.append(np.mean(c1_tmp))
        c2_hist.append(np.mean(c2_tmp))
        [x_gan, labels_input] = generate_latent_points(latent_dim,
                                                       n_batch, key)
        y_gan = -ones((n_batch, 1))
        g_loss = gan_model.train_on_batch([x_gan, labels_input], y_gan)
        g_hist.append(g_loss)
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model)
    plot_history(c1_hist, c2_hist, g_hist)

# Load data to augment
# Note the first column of the txt file must contain the Sample label.
# No prior scaling is required, the scaling procedure has been included within the present code

dataset = pd.read_csv("data.txt", header = None)
X = pd.DataFrame.to_numpy(dataset.loc[:,1:dataset.shape[1]], dtype = "float")
X_std = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
X = X_std * (1 - -1) + -1
y = dataset.loc[:,0].astype("category")

# Create and a key with categorical variable values once encoded

key = pd.DataFrame(index = range(y.cat.categories.size),
                        columns = ["key_value", "sample"])
for i in range(y.cat.categories.size):
    key.loc[i, "key_value"] = i
    key.loc[i, "sample"] = y.cat.categories[i]
    
y_values = pd.DataFrame(index = range(y.shape[0]), columns = range(1))
for i in range(y.shape[0]):
    for i2 in range(key.shape[0]):
        if y[i] == key.loc[i2, "sample"]:
            y_values.loc[i] = key.loc[i2, "key_value"]
y_values = pd.DataFrame.to_numpy(y_values, dtype = "float")

# Print the key

key

# Set hyperparameters

n_epochs = 400
n_batch = 16
latent_dim = 50

# Train GAN to augment data

c_model = define_critic_gp(X)
g_model = define_generator(X, latent_dim, key)
gan_model = define_gan(g_model, c_model, latent_dim)
start_time = time.time()
wass_train(g_model, c_model, gan_model, X,
           latent_dim = latent_dim, key, y_values, n_epoch = n_epoch, n_batch = n_batch)
end_time = time.time()
print("train_discriminator time: %s" % (end_time - start_time))
