# set directory
import os
directory = "C:/Users/" # complete directory
os.chdir(directory)

# load libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Reshape, Input, Concatenate
from tensorflow.keras.layers import LeakyReLU, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
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

# Define discriminator

def define_discriminator(dataset):
    model = Sequential()
    init = RandomNormal(stddev = 0.1)
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
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = "mse", optimizer = opt,
                 metrics = ["accuracy"])
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

def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = "mse", optimizer = opt)
    return model

# Define functions for generation of real and fake samples

def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = ones((n_samples, 1))
    return x, y
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    y = zeros((n, 1))
    return X, y

# Define functions for generation of latent point inputs

def generate_latent_points(latent_dim, n):
    x_input = rand(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

# Define function to summarize, save and checkpoint GAN performance

def summarize_performance(step, generator, discriminator,
                         latent_dim, n, dataset):
    test_real, label_real = generate_real_samples(dataset, n)
    _, acc_real = discriminator.evaluate(test_real, label_real, verbose = 0)
    test_fake, label_fake = generate_fake_samples(generator, latent_dim, n)
    _, acc_fake = discriminator.evaluate(test_fake, label_fake, verbose = 0)
    print("Step", step, " : ",
          "acc_real: ", acc_real, " ",
          "acc_fake: ", acc_fake)
    plt.scatter(dataset[:, 0], dataset[:, 1], color = "red")
    plt.scatter(test_fake[:, 0], test_fake[:, 1], color = "blue")
    title_name = "Step: %s" % step
    plt.title(title_name, loc='center')
    filename = "generated_plot_e%03d.png" % (step + 1)
    plt.savefig(filename)
    plt.close()
    h5_name = "generator_model_e%03d.h5" % (step + 1)
    generator.save(h5_name)

# Define function to plot training history

def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label = "d-real")
    plt.plot(d2_hist, label = "d-fake")
    plt.plot(g_hist, label = "gen")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(a1_hist, label = "acc-real")
    plt.plot(a2_hist, label = "acc-fake")
    plt.legend()
    plt.savefig("plot_line_plot_loss_acc.png")

# Define function to plot loss over time

def plot_loss(d1_hist, d2_hist, g_hist):
    plt.plot(d1_hist, label = "d-real")
    plt.plot(d2_hist, label = "d-fake")
    plt.plot(g_hist, label = "gen")
    plt.legend()
    plt.savefig("plot_line_plot_loss.png")

# Define final function for training

def train(dataset, g_model, d_model, gan_model, latent_dim,
            n_epochs, n_batch, n_eval = 100):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2) # only half a batch is necessary for the discriminator
    n_steps = bat_per_epo * n_epochs
    d1_hist, d2_hist, g_hist = list(), list(), list()
    a1_hist, a2_hist = list(), list()
    for i in range(n_steps):
        x_real, y_real = generate_real_samples(dataset, half_batch)
        d_loss1, d_acc1 = d_model.train_on_batch(x_real, y_real)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2, d_acc2 = d_model.train_on_batch(x_fake, y_fake)
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(x_gan, y_gan)
        print(">%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d" %
              (i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim, 100, X)
    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
    return d1_hist, d2_hist, g_hist, a1_hist, a2_hist

# Load data to augment

dataset = pd.read_csv("scaled_data.txt", header = None)
X = pd.DataFrame.to_numpy(dataset.loc[:,0:dataset.shape[1]], dtype = "float")

# Set hyperparameters

n_epochs = 400
n_batch = 16
latent_dim = 50

# Train GAN to augment data

d_model = define_discriminator(X)
g_model = define_generator(X, latent_dim = latent_dim)
gan_model = define_gan(g_model, d_model)
start_time = time.time()
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = train(X, g_model, d_model, gan_model,
                                                   latent_dim = latent_dim, n_epochs = n_epochs, n_batch = n_batch)
end_time = time.time()
print("train_discriminator time: %s" % (end_time - start_time))

# plot training history

plot_loss(d1_hist, d2_hist, g_hist)