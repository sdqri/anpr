import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
batch_size = 1024
epochs = 10
categories = len(labels)

if __name__ == "__main__":
    emnist_dataset, emnist_info = tfds.load(name='emnist', split='train', batch_size=-1,
                                            with_info=True, as_supervised=True)
    x, y = tfds.as_numpy(emnist_dataset)
    # images are transposed
    x = np.array([np.transpose(image) for image in x])
    x = x.astype("float32")
    x /= 255.0
    # x.shape = (697932, 28, 28, 1)
    y = to_categorical(y)
    # y.shape = (697932, 62)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     input_shape=(28, 28, 1),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(64,
                                     kernel_size=(3, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(categories, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # accuracy = 0.8651
    model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=1)

    model.save("ocr_model.h5")