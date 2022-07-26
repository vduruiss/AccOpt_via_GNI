# Data Fitting

'''
Given data points from a noisy version of a chosen function,
we learn the parameters of a Neural Network
to obtain a model which fits the data points as well as possible

More details can be found in
    "Practical Perspectives on Symplectic Accelerated Optimization"
    Authors: Valentin Duruisseaux and Melvin Leok. 2022.

Usage:

	python ./TensorFlow_Codes/DataFitting.py

'''


################################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import BrAVO_tf

################################################################################
# Create data corresponding to a funcction and add noise
NumberDataPoints = 500
x = np.linspace(-2, 2, num=NumberDataPoints)
y = 10*x*np.abs(np.cos(2*x)) + 10*np.exp(-np.sin(x))
y_noisy = y + 0.2*np.random.normal(size=NumberDataPoints)


################################################################################
# Create models

# Create the model with ADAM
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
model1 = keras.Sequential()
model1.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model1.add(keras.layers.Dense(units = 64, activation = 'relu'))
model1.add(keras.layers.Dense(units = 64, activation = 'relu'))
model1.add(keras.layers.Dense(units = 1, activation = 'linear'))
model1.compile(loss='mse', optimizer=optimizer1)


# Create the model with SGD
optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.001)
model2 = keras.Sequential()
model2.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model2.add(keras.layers.Dense(units = 64, activation = 'relu'))
model2.add(keras.layers.Dense(units = 64, activation = 'relu'))
model2.add(keras.layers.Dense(units = 1, activation = 'linear'))
model2.compile(loss='mse', optimizer=optimizer2)

# Create the model with eBRAVO
optimizer3 = BrAVO_tf.eBravo(learning_rate=0.07, C = 1e3)
model3 = keras.Sequential()
model3.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model3.add(keras.layers.Dense(units = 64, activation = 'relu'))
model3.add(keras.layers.Dense(units = 64, activation = 'relu'))
model3.add(keras.layers.Dense(units = 1, activation = 'linear'))
model3.compile(loss='mse', optimizer=optimizer3)

# Create the model with pBRAVO
optimizer4 = BrAVO_tf.pBravo(learning_rate=0.0007, C = 1e2)
model4 = keras.Sequential()
model4.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model4.add(keras.layers.Dense(units = 64, activation = 'relu'))
model4.add(keras.layers.Dense(units = 64, activation = 'relu'))
model4.add(keras.layers.Dense(units = 1, activation = 'linear'))
model4.compile(loss='mse', optimizer=optimizer4)



################################################################################
# Training

ADAM_losses = []
SGD_losses = []
eBrAVO_losses = []
pBrAVO_losses = []

PlottingStep = 40;  # Plot evolution every PlottingStep epochs

for k in range(50):

  # Update Models
  history1 = model1.fit( x, y_noisy, batch_size=32, epochs=PlottingStep, verbose=False)
  history2 = model2.fit( x, y_noisy, batch_size=32, epochs=PlottingStep, verbose=False)
  history3 = model3.fit( x, y_noisy, batch_size=32, epochs=PlottingStep, verbose=False)
  history4 = model4.fit( x, y_noisy, batch_size=32, epochs=PlottingStep, verbose=False)

  # Add Computed Losses
  ADAM_losses = np.append(ADAM_losses, history1.history['loss'])
  SGD_losses = np.append(SGD_losses, history2.history['loss'])
  eBrAVO_losses = np.append(eBrAVO_losses, history3.history['loss'])
  pBrAVO_losses = np.append(pBrAVO_losses, history4.history['loss'])

  # Plot models every PlottingStep epochs

  plt.subplots(figsize=(14, 7))

  plt.subplot(2, 2, 1)
  plt.scatter(x, y_noisy, color='green', s=0.5)
  plt.plot(x, model1.predict(x), 'b', linewidth= 2,label="ADAM")
  plt.ylim(bottom = 4 ,top=22)
  plt.legend()

  plt.subplot(2, 2, 2)
  plt.scatter(x, y_noisy, color='green', s=0.5)
  plt.plot(x, model2.predict(x), 'b', linewidth= 2, label="SGD")
  plt.ylim(bottom = 4 ,top=22)
  plt.legend()

  plt.subplot(2, 2, 3)
  plt.scatter(x, y_noisy, color='green', s=0.5)
  plt.plot(x, model3.predict(x), 'b', linewidth= 2, label="eBrAVO")
  plt.ylim(bottom = 4 ,top=22)
  plt.legend()

  plt.subplot(2, 2, 4)
  plt.scatter(x, y_noisy, color='green', s=0.5)
  plt.plot(x, model4.predict(x), 'b', linewidth= 2, label="pBrAVO")
  plt.ylim(bottom = 4 ,top=22)
  plt.legend()

  plt.show(block=False)
  plt.pause(0.00001)
  plt.close()


################################################################################
# Final Plots

# Plot final models

plt.subplots(figsize=(14, 7))

plt.subplot(2, 2, 1)
plt.scatter(x, y_noisy, color='green', s=0.5)
plt.plot(x, model1.predict(x), 'b', linewidth= 2,label="ADAM")
plt.ylim(bottom =4 ,top=22)
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(x, y_noisy, color='green', s=0.5)
plt.plot(x, model2.predict(x), 'b', linewidth= 2, label="SGD")
plt.ylim(bottom =4 ,top=22)
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(x, y_noisy, color='green', s=0.5)
plt.plot(x, model3.predict(x), 'b', linewidth= 2, label="eBrAVO")
plt.ylim(bottom =4 ,top=22)
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x, y_noisy, color='green', s=0.5)
plt.plot(x, model4.predict(x), 'b', linewidth= 2, label="pBrAVO")
plt.ylim(bottom =4 ,top=22)
plt.legend()
plt.tight_layout()
plt.savefig('figureA.png', bbox_inches='tight',dpi=500)

# Plot Losses

plot2 = plt.figure(2,figsize=(12, 6))
plt.plot(ADAM_losses,'black',label="ADAM")
plt.plot(SGD_losses,'green',label="SGD")
plt.plot(eBrAVO_losses,'blue',label="eBrAVO")
plt.plot(pBrAVO_losses,'red',label="pBrAVO")
plt.ylim(bottom = 0 ,top=5)
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.legend()
plt.tight_layout()
plt.savefig('figureB.png', bbox_inches='tight',dpi=500)
plt.show()
