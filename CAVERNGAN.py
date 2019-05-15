from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, UpSampling3D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

## Data Processing
WIDTH = 8
HEIGHT = 6
INPUT_SHAPE = (WIDTH, HEIGHT, WIDTH, 1)
NOISE_DIM = 100
BATCH_MOMENTUM=0.8
DROPOUT_RATE=0.25
RELU_ALPHA=0.2

opt = Adam(0.0002, 0.5)

def load_data():
    names = ["l_01.txt", "l_02.txt", "l_03.txt", "l_04.txt", "l_05.txt",
             "l_06.txt", "l_07.txt", "l_08.txt", "l_09.txt", "l_10.txt"]
    module = np.zeros((len(names), WIDTH, HEIGHT, WIDTH, 1))
    
    for w in range(len(names)):
        with open("l_modules/" + names[w], "r") as f:
            for z in range(WIDTH):
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        line = f.readline()
                        module[w, x, y, z, 0] = int(line)
    return module
                
def build_generator():
    g = Sequential()
    
    g.add(Dense(2 * 1 * 2 * 128, activation="relu", input_dim=NOISE_DIM))
    g.add(Reshape((2, 1, 2, 128)))
    
    g.add(UpSampling3D())
    g.add(Conv3D(128, kernel_size=3, padding="same"))
    g.add(BatchNormalization(momentum=BATCH_MOMENTUM))
    g.add(Activation("relu"))
    
    g.add(UpSampling3D())
    g.add(Conv3D(64, kernel_size=3, padding="same"))
    g.add(BatchNormalization(momentum=BATCH_MOMENTUM))
    g.add(Activation("relu"))
    
    g.add(UpSampling3D())
    g.add(Conv3D(1, kernel_size=(9, 3, 9), padding="valid"))
    g.add(Activation("sigmoid"))
    
    print("-"*15, "GENERATOR SUMMARY", "-"*15)
    g.summary()
    noise = Input(shape=(NOISE_DIM,))
    img = g(noise)
    
    return Model(noise, img)

def build_discriminator():
    d = Sequential()
    
    d.add(Conv3D(32, kernel_size=3, strides=1, padding="same", input_shape=INPUT_SHAPE))
    d.add(BatchNormalization(momentum=BATCH_MOMENTUM))
    d.add(LeakyReLU(alpha=RELU_ALPHA))
    d.add(Dropout(DROPOUT_RATE))
    
    d.add(Conv3D(64, kernel_size=3, strides=2, padding="same"))
    d.add(BatchNormalization(momentum=BATCH_MOMENTUM))
    d.add(LeakyReLU(alpha=RELU_ALPHA))
    d.add(Dropout(DROPOUT_RATE))
    
    d.add(Conv3D(128, kernel_size=3, strides=2, padding="same"))
    d.add(BatchNormalization(momentum=BATCH_MOMENTUM))
    d.add(LeakyReLU(alpha=RELU_ALPHA))
    d.add(Dropout(DROPOUT_RATE))
    
    d.add(Flatten())
    d.add(Dense(1, activation="sigmoid"))
    
    print("-"*15, "DISCRIMINATOR SUMMARY", "-"*15)
    d.summary()
    
    return d

def plot_loss(epoch, dLosses, gLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminative Loss')
    plt.plot(gLosses, label='Generative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/CavernGAN/gan_loss_epoch_%d.png' % epoch)

def train(gan, generator, discriminator, epochs=100, batch_size=7):
    modules = load_data()
    plot_freq = 10
    idx = 0
    
    # Setting up valid/fake data for discriminator
    valid = np.ones((batch_size, 1))
    valid[:] = 0.9
    fake = np.zeros((batch_size, 1))
    fake[:] = 0.1
    
    dLosses = []
    gLosses = []
    
    for epoch in range(epochs):
        # - Training Discriminator -
        # Get random set of images

        ## Train discriminator twice as much
        for a in range(2):
            real_imgs = modules[np.random.randint(0, modules.shape[0], batch_size)]
            
            # Set up noise and generate a batch of fake images
            noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
            fake_imgs = generator.predict(noise)
            
            for i in range(len(fake)):
                rand = np.random.randint(0, 15)
                if (rand == 0):
                    fake[i] = 0.9
                    valid[i] = 0.1
                    
            
            # Training the discriminator
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))

        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, valid)
        
        # Plots
        dLosses.append(d_loss)
        gLosses.append(g_loss)
        
        if (epoch % plot_freq == 0):            
            print("-"*15, " Epoch ", epoch, " ", "-"*15)
            #plot_loss(epoch, dLosses, gLosses)
            print("dLoss: ", d_loss)
            print("gLoss: ", g_loss)
            create_modules(generator, epoch, 3)
#%%
def create_modules(generator, epoch, num=10):
    for i in range(num):
        noise = np.random.normal(0, 1, (1, NOISE_DIM))
        outcome = generator.predict(noise)
        outcome = outcome.reshape(8, 6, 8)
        output = ""
        for z in range(WIDTH):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    output += (str(min(1, round(outcome[x, y, z], 3))))
                    output += ("\n")
                    
        with open("CavernGAN/epoch_" + str(epoch) + "_" + str(i) + ".txt", "w+") as f:
            f.write(output)
        
        f.close()
        
#%%
## Actual Script
data = load_data()
g = build_generator()
g.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

d = build_discriminator()
d.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
d.trainable=False

ganIn = Input(shape=(NOISE_DIM,))
gOut = g(ganIn)
dOut = d(gOut)
gan = Model(ganIn, dOut)
gan.compile(loss="binary_crossentropy", optimizer=opt)
print("-"*15, "GAN SUMMARY", "-"*15)
gan.summary()

train(gan, g, d, epochs=30000)

#%%

