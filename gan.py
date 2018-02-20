import os
import numpy as np
import sklearn
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.layers import Conv2DTranspose, UpSampling2D, Activation
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

par_dir = "E:/Tensorflow/notMNIST/notMNIST_small"
path = os.listdir(par_dir)
image_list = []
label=0
label_list = []
batch_size = 128

for folder in path:
    images = os.listdir(par_dir + '/' + folder)
    for image in images:
        if(os.path.getsize(par_dir +'/'+ folder +'/'+ image) > 0):
            img = cv2.imread(par_dir +'/'+ folder +'/'+ image, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_list.append(img)
            label_list.append(label)
        else:
            print('File' + par_dir +'/'+ folder +'/'+ image + 'is empty')
    label += 1


print("Looping done")

image_array = np.array(image_list, dtype=np.float32)
image_array = np.reshape(image_array, [len(image_list), 28, 28, 1])
'''
cv2.imshow('test', image_array[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
label_array = np.array(label_list)

one_hot = np.eye(10)[label_array]

image_data, one_hot = sklearn.utils.shuffle(image_array, one_hot)

print("Data ready. Bon Apetiet!")

image_train, label_train = image_data[0:12800], one_hot[0:12800]
image_test, label_test = image_data[12800:17920], one_hot[12800:17920]

def get_train_image(input):

    batch_images = image_train[(input*batch_size):((input+1)*batch_size)]
    batch_label = label_train[(input*batch_size):((input+1)*batch_size)]
    return batch_images, batch_label



# Generative Adverserial Network

epochs = 20


# Discriminator:

dropout_d = 0.4

discriminator = Sequential()

discriminator.add(Conv2D(64, 5, strides=2, activation=LeakyReLU(alpha=0.2), input_shape=(28,28,1), padding='same'))
discriminator.add(Dropout(dropout_d))

discriminator.add(Conv2D(128, 5, strides=2, activation=LeakyReLU(alpha=0.2), padding='same'))
discriminator.add(Dropout(dropout_d))

discriminator.add(Conv2D(256, 5, strides=2, activation=LeakyReLU(alpha=0.2), padding='same'))
discriminator.add(Dropout(dropout_d))

discriminator.add(Conv2D(512, 5, strides=1, activation=LeakyReLU(alpha=0.2), padding='same'))
discriminator.add(Flatten())
discriminator.add(Dropout(dropout_d))

discriminator.add(Dense(1))

discriminator.add(Activation('sigmoid'))

# Generator:

dropout_g = 0.4

generator = Sequential()

generator.add(Dense(7*7*256, input_dim=100))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Reshape((7,7,256)))
generator.add(Dropout(dropout_g))

generator.add(UpSampling2D())
generator.add(Conv2DTranspose(128, 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Dropout(dropout_g))

generator.add(UpSampling2D())
generator.add(Conv2DTranspose(64, 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Dropout(dropout_g))

generator.add(Conv2DTranspose(32, 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Dropout(dropout_g))

generator.add(Conv2DTranspose(1, 5, padding='same'))
generator.add(Activation('sigmoid'))

# Discriminator Model:

optimizer_d = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)

DM = Sequential()
DM.add(discriminator)
DM.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])

# Adverserial Model:

optimizer_a = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

AD = Sequential()
AD.add(generator)
AD.add(discriminator)
AD.compile(loss='binary_crossentropy', optimizer=optimizer_a, metrics=['accuracy'])

# Checkpoint:

filepath_DM = "E:/Tensorflow/GAN/checkpoint/weights_DM.hdf5"
filepath_AD = "E:/Tensorflow/GAN/checkpoint/weights_AD.hdf5"
checkpoint_DM = ModelCheckpoint(filepath_DM, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_AD = ModelCheckpoint(filepath_AD, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_DM = [checkpoint_DM]
callbacks_AD = [checkpoint_AD]

# Training:

for i in range(epochs):

    if os.path.isfile(filepath_DM) and os.path.isfile(filepath_AD):
        DM.load_weights("filepath_DM")
        AD.load_weights("filepath_AD")
        

    X_batch, _ = get_train_image(i)
    #X_batch = np.reshape(X_batch, [X_batch.shape[0],X_batch.shape[1],X_batch.shape[2]])

    noise = np.random.uniform(-1.0, 1.0, size=[batch_size,100])
    g_img = generator.predict(noise)

    x = np.concatenate((X_batch, g_img), axis=0)
    y = np.ones([batch_size*2, 1])

    y[batch_size:, :] = 0

    d_loss = DM.train_on_batch(x,y)

    # Adverserial Model training:

    noise_a = np.random.uniform(-1.0, 1.0, size=[batch_size,100])
    y_a = np.ones([batch_size, 1])

    g_loss = AD.train_on_batch(noise_a, y_a)

    
