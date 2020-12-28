import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

import datetime as dt

from project_dirs import _images_dir


start = dt.datetime.now()

np.random.seed(42)

# Download data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Mini-set (batch) size
batch_size = 32
# number of image classes
nb_classes = 10
# Epochs number
nb_epoch = 30
# Image size
img_rows, img_cols = 32, 32
# Number of image color channels (RGB)
img_channels = 3

# Data normalization
X_train = X_train
y_train = y_train

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Create NN model
model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3,3), padding='same',
                    input_shape=(32,32,3), activation='relu'))
# Second convolutional layer
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

# First subdiscretization layer
model.add(MaxPooling2D(pool_size=(2,2)))

# First dropout layer
model.add(Dropout(0.25))

# Third convolutional layer
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
# Fourth convolutional layer
model.add(Conv2D(64, (3,3), activation='relu'))

# Second subdiscretization layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Second dropout layer
model.add(Dropout(0.25))

# Input data converting layer
model.add(Flatten())

# Full-linked layer
model.add(Dense(32*32, activation='relu'))

# Third dropout layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(nb_classes, activation='softmax'))

# Optimization parameters
sgd = SGD(learning_rate=0.03, decay=1e-6, momentum=0.3, nesterov=True)
model.compile(loss='categorical_crossentropy', 
                optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            validation_split=0.1,
            shuffle=True,
            verbose=2)

# Estimate train quality on test data
scores = model.evaluate(X_test, Y_test, verbose=0)
end = dt.datetime.now()
print("Accuracy on test data: %.2f%%" % (scores[1]*100))
print("Time spent", end-start)

# Save model
while True:
    try:
        model.save(_images_dir+'/'+input("Enter model name\n")+".h5")
    except Exception as e:
        print("Exception occured while saving.\n")
        print("Original text:\n", e,"\n")
        print("Try again")
    else:
        input("It's OK. Press 'Enter' to close\n")
        break