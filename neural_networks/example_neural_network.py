import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# create some example data with two features

X_train = np.array([[0.1, 0.5],
             [0.3, 0.3],
             [0.2, 0.2],
             [0.8, 0.3],
             [0.9, 0.2]], dtype=np.float32)

Y_train = np.array([0,0,0,1,1], dtype=np.float32)

# set the random seed
tf.random.set_seed(1234)

# create a model with 3 dense layers
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(units=5,  activation = 'sigmoid', name='L1'),
        #tf.keras.layers.Dense(units=5,  activation = 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.1), name='L1')
        tf.keras.layers.Dense(units=3,  activation = 'sigmoid', name='L2'),
        tf.keras.layers.Dense(units=1,  activation = 'sigmoid', name='L3')
    ]
)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
model.fit(X_train, Y_train, epochs=10)
model.summary()

