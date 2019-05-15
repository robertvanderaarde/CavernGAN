from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

#%% Helper Functions
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32)) / 255 # Set to 0-1 for each pixel value
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
    
#%% Model Generation
optimizer = Adam(0.0002, 0.5)
noise_dim = 100
image_shape = (28, 28, 1)
dropout_rate = 0.25
relu_alpha = 0.2

def generator_model():
    g = Sequential()
    
    # Setting up first layer as 7x7 with 128 depth
    # Our final output is 28x28, so we need something divisible by that
    # 7x7 works because we upsample to 14x14 then 28x28
    g.add(Dense(128 * 7 * 7, activation="relu", input_dim=noise_dim))
    g.add(Reshape((7, 7, 128)))
    
    
    # First convolutional layer, shaping to 14 x 14 x 128
    g.add(UpSampling2D())
    g.add(Conv2D(128, kernel_size=3, padding="same"))
    g.add(BatchNormalization(momentum=0.8))
    g.add(Activation("relu"))
    
    # Second convolutional layer, shaping to 28x28x64
    g.add(UpSampling2D())
    g.add(Conv2D(64, kernel_size=3, padding="same"))
    g.add(BatchNormalization(momentum=0.8))
    g.add(Activation("relu"))
    
    # Third layer, shaping to 28x28x1 with inputs to 0->1
    g.add(Conv2D(1, kernel_size=3, padding="same"))
    g.add(Activation("sigmoid"))
    
    g.summary()
    noise = Input(shape=(noise_dim,))
    img = g(noise)
    
    return Model(noise, img)

def discriminator_model():
    d = Sequential()
    
    # In the discriminator, we need to reshape our inputs from 28x28x1 to a deep
    # but small width/height convolutional network, then have a single output with
    # fake or real
    
    # Instead of MaxPooling, we use strides to shorten the width and height, this
    # has a better outcome by using every pixel instead of just generalizing
    # Creates a 14x14x32 conv layer
    d.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    d.add(LeakyReLU(alpha=relu_alpha))
    d.add(Dropout(dropout_rate))
    
    # Not sure why we use this ZeroPadding, maybe toy with removing it
    d.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    d.add(ZeroPadding2D(padding=((0,1),(0,1))))
    d.add(BatchNormalization(momentum=0.8))
    d.add(LeakyReLU(alpha=relu_alpha))
    d.add(Dropout(dropout_rate))
    
    d.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    d.add(BatchNormalization(momentum=0.8))
    d.add(LeakyReLU(alpha=relu_alpha))
    d.add(Dropout(dropout_rate))
    
    d.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    d.add(BatchNormalization(momentum=0.8))
    d.add(LeakyReLU(alpha=relu_alpha))
    d.add(Dropout(dropout_rate))
    
    # Flatten to singular Dense output
    d.add(Flatten())
    d.add(Dense(1, activation="sigmoid"))
    
    d.summary()
    
    return d

#%% Setting up generator and discriminator
generator = generator_model()
generator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Can we just set this later?
discriminator.trainable=False

# Setting up the actual gan model
ganIn = Input(shape=(noise_dim,))
gOut = generator(ganIn)
dOut = discriminator(gOut)
gan = Model(ganIn, dOut)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)
gan.summary()

def train(gan, generator, discriminator, epochs=100, batch_size=5):
    X_train, _, _, _ = load_data()
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_train = X_train[:25]
    plot_freq = 1000
    
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
        real_imgs = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        
        # Set up noise and generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_imgs = generator.predict(noise)
        for i in range(len(fake)):
            rand = np.random.randint(0, 25)
            if (rand == 0):
                fake[i] = 0.9
                valid[i] = 0.1
                
        
        # Training the discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # Train the generator
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, valid)
        
        # Plots
        dLosses.append(d_loss)
        gLosses.append(g_loss)
        
        if (epoch % plot_freq == 0):            
            print("-"*15, " Epoch ", epoch, " ", "-"*15)
            plot_loss(epoch, dLosses, gLosses)
            plot_images(epoch, generator)
            print("dLoss: ", d_loss)
            print("gLoss: ", g_loss)
            
    

train(gan, generator, discriminator, epochs=50000)


    
    
    
    