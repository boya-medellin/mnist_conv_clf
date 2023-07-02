import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt 
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2

IMAGES_PATH = f"{os.getcwd()}/images"

def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = f"{IMAGES_PATH}/{fig_id}.{fig_extension}"
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_history(history, epochs: int, title: str) -> None:
    pd.DataFrame(history.history).plot(
            figsize=(8, 5), xlim=[0, epochs], ylim=[0,1], grid=True, xlabel='Epoch',
            style=['r--', 'r--', 'b-', 'b-*'])
    plt.legend(loc='lower left')
    plt.title(title)
    save_fig('history')
    plt.show()

def show_sample(X, y, n_rows=4, n_cols=10 ):
    class_names = np.arange(0, 10)

    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))

    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X[index], cmap='binary', interpolation='nearest')
            plt.axis('off')
            plt.title(class_names[y[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    save_fig('sample')
    plt.show()

# Load Data
mnist = tf.keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

#show_sample(X_train, y_train)

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Reshape([28, 28, 1]),
    tf.keras.layers.Conv2D(filters=64, 
                            kernel_size=5, 
                            padding='same', 
                            activation='relu', 
                            kernel_regularizer=l2(0.1),
                            input_shape=[28,28,1]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=128, 
                            kernel_size=3, 
                            padding='same', 
                            activation='relu', 
                            kernel_regularizer=l2(0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=128, 
                            kernel_size=3, 
                            padding='same', 
                            activation='relu', 
                            kernel_regularizer=l2(0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Nadam()
epochs=1

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# Train
X_train = X_train[:1000]
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
# plot_history(history, epochs, 'title')
# score = model.evaluate(X_test, y_test)
