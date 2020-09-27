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
    model = Sequential()
    model.add(Dense(math.ceil((1/2)*dataset.shape[1]),
                    kernel_initializer = init,
                    input_dim = dataset.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dense(math.ceil((1/4)*dataset.shape[1]),
                    kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = "linear"))
    opt = RMSprop(lr = 0.00005)
    model.compile(loss = wasserstein_loss, optimizer = opt)
    return model

# Define generator

def define_generator(dataset, latent_dim):
    model = Sequential()
    init = RandomNormal(stddev = 0.7)
    model.add(Dense(math.ceil((1/4)*dataset.shape[1]),
                    kernel_initializer = init,
                    input_dim = latent_dim))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dense(math.ceil((1/2)*dataset.shape[1]),
                    kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dense((dataset.shape[1]+math.ceil((1/4)*dataset.shape[1])),
                     kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dense(dataset.shape[1], activation = "tanh"))
    return model

# Define GAN

def define_gan(generator, critic):
    critic.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(critic)
    opt = RMSprop(lr = 0.00005)
    model.compile(loss = wasserstein_loss, optimizer = opt)
    return model

# Define functions for generation of real and fake samples

def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = -ones((n_samples, 1))
    return x, y
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    y = ones((n, 1))
    return X, y

# Define functions for generation of latent point inputs

def generate_latent_points(latent_dim, n):
    x_input = rand(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

# Define function to summarize, save and checkpoint GAN performance

def summarize_performance(step, generator, discriminator,
                         latent_dim, n, dataset):
    new, _ = generate_fake_samples(generator, latent_dim, n)   
    plt.scatter(new[:, 0], new[:, 1], color = "blue")
    plt.scatter(dataset[:, 0], dataset[:, 1], color = "red")
    title_name = "Step: %s" % step
    plt.title(title_name, loc='center')
    filename = "generated_plot_e%03d.png" % (step + 1)
    plt.savefig(filename)
    plt.close()
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

def wass_train(g_model, c_model, gan_model, dataset, latent_dim,
              n_epoch, n_batch, n_critic = 5, n_eval = 100):
    bat_per_epo = int(dataset.shape[0]/n_batch)
    n_steps = bat_per_epo * n_epoch
    half_batch = int(n_batch/2)
    c1_hist, c2_hist, g_hist = list(), list(), list()
    for i in range(n_steps):
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            x_real, y_real = generate_real_samples(dataset, half_batch)
            c_loss1 = c_model.train_on_batch(x_real, y_real)
            c1_tmp.append(c_loss1)
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            c_loss2 = c_model.train_on_batch(x_fake, y_fake)
            c2_tmp.append(c_loss2)
        c1_hist.append(np.mean(c1_tmp))
        c2_hist.append(np.mean(c2_tmp))
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = -ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(x_gan, y_gan)
        g_hist.append(g_loss)
        print(">%d, c1=%.3f, c2=%.3f g=%.3f" %
              (i+1, c1_hist[-1], c2_hist[-1], g_loss))
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, c_model, latent_dim, 100, X)
    plot_history(c1_hist, c2_hist, g_hist)

# Load data to augment

dataset = pd.read_csv("scaled_data.txt", header = None)
X = pd.DataFrame.to_numpy(dataset.loc[:,0:dataset.shape[1]], dtype = "float")

# Set hyperparameters

n_epochs = 400
n_batch = 16
latent_dim = 50

# Train GAN to augment data

c_model = define_critic_gp(X)
g_model = define_generator(X, latent_dim)
gan_model = define_gan(g_model, c_model)
start_time = time.time()
wass_train(g_model, c_model, gan_model, X,
           latent_dim = latent_dim, n_epoch = n_epoch, n_batch = n_batch)
end_time = time.time()
print("train_discriminator time: %s" % (end_time - start_time))