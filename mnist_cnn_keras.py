import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# load the data
# image size 28x28 px, output 1 int
# 60,000 training data, 10,000 test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the pixel values
X_train /= 255
X_test /= 255


# reshape the output
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# let there be model
model = Sequential([
	Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
	Conv2D(64, (3, 3), activation='relu'),
	MaxPooling2D(pool_size=(2, 2)),
	Dropout(0.25),
	Flatten(),
	Dense(128, activation='relu'),
	Dropout(0.5),
	Dense(10, activation='softmax')
])

# define criteria
model.compile(
	loss='categorical_crossentropy',
	optimizer='adadelta',
	metrics=['accuracy'])

# train
batch_size = 500
epochs = 5
model.fit(
	X_train, Y_train, 
	batch_size=batch_size, nb_epoch=epochs,
	validation_data=(X_test, Y_test),
	verbose=1)

# evaluate
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test accuracy: ', score[1])
