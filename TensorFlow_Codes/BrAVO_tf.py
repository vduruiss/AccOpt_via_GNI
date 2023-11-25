# eBrAVO and pBrAVO Algorithms within TensorFlow
'''
 Implementation of the eBrAVO and pBrAVO algorithms from
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
     Authors: Valentin Duruisseaux and Melvin Leok. 
'''


import tensorflow.compat.v2 as tf
from keras.optimizer_v2 import optimizer_v2
import numpy as np


class eBravo(optimizer_v2.OptimizerV2):
    """ Implementation of the Exponential BrAVO algorithm.

        For further details regarding the algorithm we refer to
         "Practical Perspectives on Symplectic Accelerated Optimization"
         Authors: Valentin Duruisseaux and Melvin Leok. 2022.

        Args:

            lr (float): learning rate
            C (float, optional): constant in the algorithm  (default: 1),
                                 try several values ranging from 1e-5 to 1e5
            eta (float, optional): exponential rate (default: 0.01)
                                 usually not necessary to tune
            beta (float, optional): temporal looping constant  (default: 0.8)

    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=1, C = 1, eta = 0.01, beta = 0.8,
                name='eBravo', **kwargs):
        super(eBravo, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('C', C)
        self._set_hyper('eta', eta)
        self._set_hyper('beta', beta)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'momentum')
        for var in var_list:
            self.add_slot(var, 'time_var')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(eBravo, self)._prepare_local(var_device, var_dtype, apply_state)

        apply_state[(var_device, var_dtype)].update(dict(
                                lr=self._get_hyper('learning_rate', var_dtype),
                                eta = self._get_hyper('eta', var_dtype),
                                C = self._get_hyper('C', var_dtype),
                                beta = self._get_hyper('beta', var_dtype))
                                )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))


        # Extract the momentum and time variable
        momentum = self.get_slot(var, 'momentum')
        time_var = self.get_slot(var, 'time_var')

        # Extract the hyperparameters
        eta = coefficients['eta']
        C = coefficients['C']
        beta = coefficients['beta']
        lr = coefficients['lr']

        # If it is the first iteration, set the time variable to 1
        tf.cond(tf.equal(tf.reduce_sum(time_var),0) ,
                true_fn=lambda: tf.compat.v1.assign(time_var, tf.ones_like(time_var), use_locking=self._use_locking),
                false_fn=lambda: tf.compat.v1.assign(time_var, time_var, use_locking=self._use_locking))

        # Update the momentum variable
        tf.compat.v1.assign_add(momentum, -C*lr*eta*tf.math.exp(-2*eta*time_var)*grad,
                                    use_locking=self._use_locking)

        # Update for the Position Variable
        Delta = lr*eta*tf.math.exp(-eta*(time_var+0.5*lr))*momentum

        # Momentum Restarting with the Gradient Scheme:
        # If the inner product of the gradient of f and Delta is >0
        # then restart the momentum to 0
        tf.cond( tf.greater(tf.reduce_sum(tf.math.multiply(grad,Delta)) , 0) ,
                true_fn=lambda: tf.compat.v1.assign(momentum, tf.zeros_like(momentum),
                                                    use_locking=self._use_locking),
                false_fn=lambda: tf.compat.v1.assign(momentum, momentum,
                                                    use_locking=self._use_locking)
                )

        # Temporal Looping:
        # Time Loop whenever numerical instability may be near
        qt = tf.reshape(time_var,[-1])
        tf.cond( tf.greater(C*(lr**2)*(eta**2)*tf.math.exp(eta*(qt[0]+lr))*tf.norm(grad) , tf.norm(Delta)) ,
                true_fn=lambda: tf.compat.v1.assign(time_var, tf.maximum(0.001*tf.ones_like(time_var),beta*time_var),
                                                    use_locking=self._use_locking),
                false_fn=lambda: tf.compat.v1.assign(time_var, time_var,
                                                    use_locking=self._use_locking)
                )

        # Update Time Variable
        tf.compat.v1.assign_add(time_var, lr*tf.ones_like(time_var), use_locking=self._use_locking)

        # Update Position Variable
        return tf.compat.v1.assign(var, var + Delta, use_locking=self._use_locking).op

    def get_config(self):
        config = super(eBravo, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'eta': self._serialize_hyperparameter('eta'),
            'beta': self._serialize_hyperparameter('beta'),
            'C': self._serialize_hyperparameter('C')
        })

        return config




class pBravo(optimizer_v2.OptimizerV2):
    """ Implements the Polynomial BrAVO algorithm.

        For further details regarding the algorithm we refer to
         "Practical Perspectives on Symplectic Accelerated Optimization"
         Authors: Valentin Duruisseaux and Melvin Leok. 2022.

        Args:

            lr (float): learning rate
            C (float, optional): constant in the algorithm  (default: 0.1),
                                 try several values ranging from 1e-5 to 1e5
            p (float, optional): polynomial rate (default: 6)
                                 usually not necessary to tune
            beta (float, optional): temporal looping constant  (default: 0.8)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=0.01, C = 0.1, p = 6, beta = 0.8,
                name='pBravo', **kwargs):
        super(pBravo, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('C', C)
        self._set_hyper('p', p)
        self._set_hyper('beta', beta)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'momentum')
        for var in var_list:
            self.add_slot(var, 'time_var')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(pBravo, self)._prepare_local(var_device, var_dtype, apply_state)

        apply_state[(var_device, var_dtype)].update(dict(
                                lr=self._get_hyper('learning_rate', var_dtype),
                                p = self._get_hyper('p', var_dtype),
                                C = self._get_hyper('C', var_dtype),
                                beta = self._get_hyper('beta', var_dtype))
                                )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))


        # Extract the momentum and time variable
        momentum = self.get_slot(var, 'momentum')
        time_var = self.get_slot(var, 'time_var')

        # Extract the hyperparameters
        p = coefficients['p']
        C = coefficients['C']
        beta = coefficients['beta']
        lr = coefficients['lr']

        # If it is the first iteration, set the time variable to 1
        tf.cond(tf.equal(tf.reduce_sum(time_var),0) ,
                true_fn=lambda: tf.compat.v1.assign(time_var, tf.ones_like(time_var), use_locking=self._use_locking),
                false_fn=lambda: tf.compat.v1.assign(time_var, time_var, use_locking=self._use_locking))

        # Update the momentum variable
        tf.compat.v1.assign_add(momentum, -C*lr*p*tf.math.pow(time_var, (2*p-1)*tf.ones_like(time_var))*grad,
                                    use_locking=self._use_locking)

        # Update for the Position Variable
        Delta = lr*p*tf.math.divide(momentum ,  tf.math.pow(time_var + 0.5*lr*tf.ones_like(time_var) , (p+1)*tf.ones_like(time_var)))

        # Momentum Restarting with the Gradient Scheme:
        # If the inner product of the gradient of f and Delta is >0
        # then restart the momentum to 0

        tf.cond( tf.greater(tf.reduce_sum(tf.math.multiply(grad,Delta)) , 0) ,
                true_fn=lambda: tf.compat.v1.assign(momentum, tf.zeros_like(momentum),
                                                    use_locking=self._use_locking),
                false_fn=lambda: tf.compat.v1.assign(momentum, momentum,
                                                    use_locking=self._use_locking)
                )

        # Temporal Looping:
        # Time Loop whenever numerical instability may be near
        #tf.greater(C*(lr**2)*(p**2)* tf.math.pow(time_var, (p+1)*tf.ones_like(time_var)) * tf.norm(grad) ,
        #                    time_var*tf.norm(Delta)
        qt = tf.reshape(time_var,[-1])
        tf.cond( tf.greater( C*(lr**2)*(p**2)* tf.math.pow(qt[0], p+1) * tf.norm(grad)  ,
                             qt[0]*tf.norm(Delta)
                            ),
                true_fn=lambda: tf.compat.v1.assign(time_var, tf.maximum(0.001*tf.ones_like(time_var),beta*time_var),
                                                    use_locking=self._use_locking),
                false_fn=lambda: tf.compat.v1.assign(time_var, time_var,
                                                    use_locking=self._use_locking)
                )

        # Update Time Variable
        tf.compat.v1.assign_add(time_var, lr*tf.ones_like(time_var), use_locking=self._use_locking)

        # Update Position Variable
        return tf.compat.v1.assign(var, var + Delta, use_locking=self._use_locking).op


    def get_config(self):
        config = super(pBravo, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'p': self._serialize_hyperparameter('p'),
            'beta': self._serialize_hyperparameter('beta'),
            'C': self._serialize_hyperparameter('C')
        })

        return config
