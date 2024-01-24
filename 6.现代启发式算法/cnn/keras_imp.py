import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from conv3x3 import Conv3x3
from maxpool2 import MaxPool2
from softmax import Softmax
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
 
# 加载MNIST数据集
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
 
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)
 
model = Sequential([
    Conv2D(8, 3, input_shape=(28, 28, 1), use_bias=False),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(10, activation='softmax'),
])
 
model.compile(SGD(lr=.005), loss='categorical_crossentropy', metrics=['accuracy'])
 
model.fit(
    train_images,
    to_categorical(train_labels),
    batch_size=1,
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
)