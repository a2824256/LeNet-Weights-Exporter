import keras
from keras.datasets import mnist
from keras.layers import Conv2D, AvgPool2D, Dense, Flatten
from keras.models import Sequential

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# CPU
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
# if your want to use gpu
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train / 255.
x_test = x_test / 255.

x_train = x_train.reshape(-1, 28, 28, 1)
x_test=x_test.reshape(-1,28,28,1)

lenet = Sequential()
lenet.add(Conv2D(6, kernel_size=5, strides=1, padding='same', input_shape=(28, 28, 1)))
lenet.add(AvgPool2D(pool_size=2, strides=2))
lenet.add(Conv2D(16, kernel_size=5, strides=1, padding='valid'))
lenet.add(AvgPool2D(pool_size=2, strides=2))
lenet.add(Flatten())
lenet.add(Dense(120))
lenet.add(Dense(84))
lenet.add(Dense(10, activation='softmax'))


lenet.summary()


lenet.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])

lenet.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=[x_test, y_test])
weights = lenet.get_weights()
lenet.save('myletnet.h5')
