import gpflow

import tensorflow as tf
from functools import reduce

class Brownian(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(input_dim=1, active_dims=[0])
        self.variance = gpflow.Param(1.0, transform=gpflow.transforms.positive)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance * tf.minimum(X, tf.transpose(X2))

    def Kdiag(self, X):
        return self.variance * tf.reshape(X, (-1,))
    
# https://github.com/GPflow/GPflow/issues/453
class ChangePoint(gpflow.kernels.Combination):
    
    def __init__(self, kern_list, offset, variance):
        super(ChangePoint, self).__init__(kern_list)
        self.offset = gpflow.Param(offset)
        self.variance = gpflow.Param(variance)

    def K(self, X, X2=None, presliced=False):
        # assert len(self.kern_list) != 3, "has to have two kernels only"
        if X2 is None:
            X2 = X
        XXT = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X, axis=1), tf.expand_dims(X, axis=2))) + self.offset
        XXT2 = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X2, axis=1), tf.expand_dims(X2, axis=2))) + self.offset
        sigm = tf.sigmoid(XXT)
        sigm2 = tf.sigmoid(XXT2)
        sig1 = tf.matmul(tf.expand_dims(sigm, axis=1),
                         tf.expand_dims(sigm2, axis=0))
        sig2 = tf.matmul(tf.expand_dims((1. - sigm), axis=1),
                         tf.expand_dims((1. - sigm2), axis=0))
        a1 = reduce(tf.multiply,
                    [sig1, self.kern_list[0].K(X, X2)])
        a2 = reduce(tf.multiply,
                    [sig2, self.kern_list[1].K(X, X2)])
        return reduce(tf.add, [a1, a2])


    def Kdiag(self, X, presliced=False):
        # assert len(self.kern_list) != 3, "has to have two kernels only"
        XXT = self.variance * tf.squeeze(tf.matmul(tf.expand_dims(X, axis=1), tf.expand_dims(X, axis=2))) + self.offset
        sigm = tf.sigmoid(XXT)
        a1 = reduce(tf.multiply, [sigm, self.kern_list[0].Kdiag(X),
                             sigm])
        a2 = reduce(tf.multiply, [(1. - sigm), self.kern_list[1].Kdiag(X),
                             1. - sigm])
        return reduce(tf.add, [a1, a2])
    
import numpy as np
kernel = ChangePoint(kern_list=[gpflow.kernels.Matern12(input_dim=1)])#
xs = np.arange(0, 100, 0.1)

print(xs)
