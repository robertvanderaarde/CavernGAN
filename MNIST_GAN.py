 ## Imports
import keras
from keras import backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2DTranspose, Conv2D
from keras.layers.core import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation, Input
from keras.layers import LeakyReLU
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adagrad, Adam
import matplotlib.pyplot as plt

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32)) / 255 # Set to 0-1 for each pixel value
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    return X_train, y_train, X_test, y_test

def plot_loss(epoch, dLosses, gLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminative Loss')
    plt.plot(gLosses, label='Generative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/mnist_gan/gan_loss_epoch_%d.png' % epoch)
    
def plot_images(epoch, generator, examples=100):
    dim=(10, 10)
    noise = np.random.normal(0, 1, size=[examples, 100])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)
    
    plt.figure(figsize=(10, 10))
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/mnist_gan/generator_image_epoch_%d.png' % epoch)

adamg = Adam(lr=0.0002, beta_1=0.5)
adamd = Adam(lr=0.0002, beta_1=0.5)


g = Sequential()
g.add(Dense(256, input_dim=100, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
#g.add(BatchNormalization())
g.add(LeakyReLU(alpha=0.2))
g.add(Dense(512))
g.add(LeakyReLU(alpha=0.2))
g.add(Dense(600))
g.add(LeakyReLU(alpha=0.2))
g.add(Dense(700))
g.add(LeakyReLU(alpha=0.2))
g.add(Dense(800))
#g.add(BatchNormalization())
g.add(LeakyReLU(alpha=0.2))
g.add(Dense(1024))
#g.add(BatchNormalization())
g.add(LeakyReLU(alpha=0.2))
g.add(Dense(784))
g.compile(loss='binary_crossentropy', optimizer=adamg, metrics=['accuracy'])

d = Sequential()
d.add(Dense(1024, input_dim=784, activation=LeakyReLU(alpha=0.2), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
d.add(Dropout(0.1))
d.add(Dense(800, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.1))
d.add(Dense(700, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.1))
d.add(Dense(600, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.1))
d.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.1))
d.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
d.add(Dropout(0.1))
d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', optimizer=adamd, metrics=['accuracy'])
d.trainable = False

inputs = Input(shape=(100,))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adamg, metrics=['accuracy'])

X_train, y_train, X_test, y_test = load_data()
X_train = X_train[np.random.randint(0, X_train.shape[0], size=25)]
print(y_train)

def train(epochs=1, plt_frq=1, batch_size=1):
    batchCount = int(X_train.shape[0] / batch_size)
    
    losses = {"D":[], "G":[]}
    d_loss = (0, 0)
    g_loss = (0, 0)
    for epoch in range(1, epochs + 1):
        if (epoch % 50 == 0):
            print('-'*15, 'Epoch %d' % epoch, '-'*15)
        for _ in range(batchCount):
            # Create a batch by drawing random numbers from training set
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            
            # Noise for the generator
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            
            # Generate images
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))
            y = np.zeros(2 * batch_size)
            y[:batch_size] = 0.9
            y[batch_size:] = 0.1
            
            rand = np.random.randint(0, 10)
            if (rand == 0):
                y[:batch_size] = 0.1
                y[batch_size + 1:] = 0.9
            
            # Train discriminator
            d.trainable = True
            d_loss = d.train_on_batch(X, y)
            
            # Train generator
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            y2 = np.ones(batch_size)
            d.trainable = False;
            g_loss = gan.train_on_batch(noise, y2)
            
        
        # Loss on final batch of epoch
        losses["D"].append(d_loss[0])
        losses["G"].append(g_loss[0])
        if (epoch == 1 or epoch % 50 == 0):
            plot_loss(epoch, losses["D"], losses["G"])
            plot_images(epoch, g)
            

train(epochs=20000)
            
            
            
            