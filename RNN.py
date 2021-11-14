import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adagrad',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Evaluating
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Save the model
# model.save('myRNN.model')

# Load in the model
# new_model = tf.keras.models.load_model('myRNN.model')

# Make a prediction using the x_test files
predictions = model.predict([x_test])

# Plot
x = 999
print(np.argmax(predictions[x]))
plt.imshow(x_test[x])
