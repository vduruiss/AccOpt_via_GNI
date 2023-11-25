# Binary Classification

'''
We test the eBrAVO and pBrAVO algorithms for Binary Classification

More details can be found in
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
     Authors: Valentin Duruisseaux and Melvin Leok. 

Usage:

	python ./TensorFlow_Codes/BinaryClassification.py

'''

################################################################################

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib
import BrAVO_tf

################################################################################

# Number of Iterations
Its = 60

# Number of Points of Each class
Samples = [250,250]

################################################################################
# Create a classification dataset for points in 2D

X, y = make_blobs(n_samples=Samples,center_box=[-2, 2] , cluster_std = 0.5)
y = 2*y-1   # Use labels {-1,1} instead of {0,1}
X_tensor = tf.convert_to_tensor(X)
y_tensor = tf.cast(tf.convert_to_tensor(y),tf.float64)



#################################################################################
# With Adam

opt = tf.keras.optimizers.Adam(learning_rate=0.2)

w = tf.Variable(np.ones(2))

loss1 = lambda: tf.reduce_sum(tf.math.log(1+tf.math.exp(tf.multiply(tf.negative(y_tensor),tf.reduce_sum(tf.multiply(w, X_tensor),1)))))
'''
loss is essentially be defined as follows:

def myLossFunction(x):
    wTx = tf.reduce_sum(tf.multiply(w, X_tensor),1)
    ywTx = tf.multiply(y_tensor,wTx)
    expo = tf.math.exp(tf.negative(ywTx))
    logterm = tf.math.log(1+expo)
    return tf.reduce_sum(logterm)
'''

ADAM_losses = [loss1().numpy()]

for k in range(Its):
    opt.minimize(loss1, [w]).numpy()
    ADAM_losses.append(loss1().numpy())

#################################################################################
# With SGD

opt = tf.keras.optimizers.SGD(learning_rate=0.05)

w = tf.Variable(np.ones(2))

loss2 = lambda: tf.reduce_sum(tf.math.log(1+tf.math.exp(tf.multiply(tf.negative(y_tensor),tf.reduce_sum(tf.multiply(w, X_tensor),1)))))
''' Same as loss1 '''

SGD_losses = [loss2().numpy()]

for k in range(Its):
    opt.minimize( loss2, [w]).numpy()
    SGD_losses.append(loss2().numpy())

#################################################################################
# With eBRAVO

opt = BrAVO_tf.eBravo(learning_rate=0.3, C = 1e3)

w = tf.Variable(np.ones(2))

loss3 = lambda: tf.reduce_sum(tf.math.log(1+tf.math.exp(tf.multiply(tf.negative(y_tensor),tf.reduce_sum(tf.multiply(w, X_tensor),1)))))
''' Same as loss1 '''

eBrAVO_losses = [loss3().numpy()]

for k in range(Its):
    opt.minimize( loss3, [w]).numpy()
    eBrAVO_losses.append(loss3().numpy())

#################################################################################
# With pBRAVO

opt = BrAVO_tf.pBravo(learning_rate=0.01)

w = tf.Variable(np.ones(2))

loss4 = lambda: tf.reduce_sum(tf.math.log(1+tf.math.exp(tf.multiply(tf.negative(y_tensor),tf.reduce_sum(tf.multiply(w, X_tensor),1)))))
''' Same as loss1 '''

pBrAVO_losses = [loss4().numpy()]

for k in range(Its):
    opt.minimize( loss4, [w]).numpy()
    pBrAVO_losses.append(loss4().numpy())



#################################################################################
# Plot dataset

x1_positive = [X[i][0].tolist() for i in range(len(y)) if y[i] == 1]
x2_positive = [X[i][1].tolist() for i in range(len(y)) if y[i] == 1]
x1_negative = [X[i][0].tolist() for i in range(len(y)) if y[i] == -1]
x2_negative = [X[i][1].tolist() for i in range(len(y)) if y[i] == -1]

plot = plt.figure(figsize=(14, 4))
grid = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])

plt.subplot(grid[0])
plt.scatter(x1_positive,x2_positive, c='blue', s=15)
plt.scatter(x1_negative,x2_negative, c='red', s=15)

# Plot Loss
plt.subplot(grid[1])
plt.plot(ADAM_losses,'black',label="ADAM", linewidth=2)
plt.plot(SGD_losses,'green',label="SGD", linewidth=2)
plt.plot(eBrAVO_losses,'blue',label="eBrAVO", linewidth=2)
plt.plot(pBrAVO_losses,'red',label="pBrAVO", linewidth=2)
plt.ylabel("Loss",fontsize=14)
plt.xlabel("Iterations",fontsize=14)
plt.yscale("log")
plt.legend(fontsize=14)


plt.tight_layout()
plt.savefig('figure.png', bbox_inches='tight',dpi=500)
plt.show()
