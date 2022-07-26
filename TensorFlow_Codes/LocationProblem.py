# Fermat-Weber Location Problem

'''
We test the eBrAVO and pBrAVO algorithms on a Fermat-Weber Location Problem

More details can be found in
    "Practical Perspectives on Symplectic Accelerated Optimization"
    Authors: Valentin Duruisseaux and Melvin Leok. 2022.
Usage:

	python ./TensorFlow_Codes/LocationProblem.py

'''

################################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import BrAVO_tf

################################################################################

# Dimension
d = 1000

# Number of Vectors y
m = 5000

# Number Iterations
Its = 30


################################################################################
# Create Dataset

# Create m Random Vectors y of size d
Y = tf.convert_to_tensor(-10+20*np.random.rand(m,d))

# Create a random vectors w of weights of size m
w = tf.convert_to_tensor(np.random.rand(m))

# Initialise x
x = tf.Variable(np.ones(d))


#################################################################################
# With Adam

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

x = tf.Variable(np.ones(d))

# Fermat-Weber Loss Function
loss1 = lambda: tf.reduce_sum(tf.math.multiply(w,tf.norm(Y - tf.reshape(tf.tile(x,tf.constant([m])),[m,d]), axis=1)))
'''
loss is essentially defined as follows:

def myLossFunction(x):
    x_2d = tf.reshape(tf.tile(x,tf.constant([m])),[m,d])
    distances_tensor = tf.norm(Y - x_2d, axis=1)
    weigthed_distances = tf.math.multiply(w,distances_tensor)
    return tf.reduce_sum(weigthed_distances)
'''

ADAM_losses = [loss1().numpy()]

for k in range(Its):
    opt.minimize(loss1, [x]).numpy()
    ADAM_losses.append(loss1().numpy())

#################################################################################
# With SGD

opt = tf.keras.optimizers.SGD(learning_rate=0.02)

x = tf.Variable(np.ones(d))

# Fermat-Weber Loss Function
loss2 = lambda: tf.reduce_sum(tf.math.multiply(w,tf.norm(Y - tf.reshape(tf.tile(x,tf.constant([m])),[m,d]), axis=1)))
''' Same as loss1 '''

SGD_losses = [loss2().numpy()]

for k in range(Its):
    opt.minimize( loss2, [x]).numpy()
    SGD_losses.append(loss2().numpy())

#################################################################################
# With eBRAVO

opt = BrAVO_tf.eBravo(learning_rate=8)
x = tf.Variable(np.ones(d))

# Fermat-Weber Loss Function
loss3 = lambda: tf.reduce_sum(tf.math.multiply(w,tf.norm(Y - tf.reshape(tf.tile(x,tf.constant([m])),[m,d]), axis=1)))
''' Same as loss1 '''

eBrAVO_losses = [loss3().numpy()]

for k in range(Its):
    opt.minimize( loss3, [x]).numpy()
    eBrAVO_losses.append(loss3().numpy())

#################################################################################
# With pBRAVO

opt = BrAVO_tf.pBravo(learning_rate=0.05)
x = tf.Variable(np.ones(d))

# Fermat-Weber Loss Function
loss4 = lambda: tf.reduce_sum(tf.math.multiply(w,tf.norm(Y - tf.reshape(tf.tile(x,tf.constant([m])),[m,d]), axis=1)))
''' Same as loss1 '''

pBrAVO_losses = [loss4().numpy()]

for k in range(Its):
    opt.minimize( loss4, [x]).numpy()
    pBrAVO_losses.append(loss4().numpy())


#################################################################################
# Plot Loss

plot1 = plt.figure(figsize=(12, 5))
plt.plot(ADAM_losses,'black',label="ADAM", linewidth=2)
plt.plot(SGD_losses,'green',label="SGD", linewidth=2)
plt.plot(eBrAVO_losses,'blue',label="eBrAVO", linewidth=2)
plt.plot(pBrAVO_losses,'red',label="pBrAVO", linewidth=2)
plt.ylabel("Loss",fontsize=14)
plt.xlabel("Iterations",fontsize=14)
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig('figure.png', bbox_inches='tight',dpi=500)
plt.show()
