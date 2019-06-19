print("synthetic.1")
import os
print("synthetic.2")
import random 
print("synthetic.3")
import sys
print("synthetic.4")
from gp_sampling.learning.kernels import Brownian
print("synthetic.5")
from gp_sampling.learning.learners import ActiveLearner
print("synthetic.6")
import numpy as np
print("synthetic.7")
import gpflow

def create_signal():
    cps = np.random.choice(list(range(1000)), size=random.randint(1, 25))
    sig = [0]
    for i in range(999):
        s = sig[-1]
        if i in cps:
            s += np.random.normal(0, 3)
        s += 0.3 * np.random.normal(0, 1)
        sig.append(s)
    return np.array(sig), cps


if __name__ == "__main__":
    for i in range(1000):
        signal, cps = create_signal()
        
        MAXITER = 100
        
        a = ActiveLearner(np.arange(0, 1000), signal, Brownian())
        means, variance = a.iterative_train(max_iter=MAXITER)
        np.savez_compressed(
            "/home/stefan/{}_Brownian".format(i),
            means=means,
            variance=variance,
            signal=signal,
            cps=cps
        )
        
        b = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.Matern32(input_dim=1))
        means, variance = b.iterative_train(max_iter=MAXITER)
        np.savez_compressed(
            "/home/stefan/{}_Matern32".format(i),
            means=means,
            variance=variance,
            signal=signal,
            cps=cps
        )
        
        c = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.Matern52(input_dim=1))
        means, variance = c.iterative_train(max_iter=MAXITER)
        np.savez_compressed(
            "/home/stefan/{}_Matern52".format(i),
            means=means,
            variance=variance,
            signal=signal,
            cps=cps
        )
        
        d = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.RBF(input_dim=1))
        means, variance = d.iterative_train(max_iter=MAXITER)
        np.savez_compressed(
            "/home/stefan/{}_RBF".format(i),
            means=means,
            variance=variance,
            signal=signal,
            cps=cps
        )
        
        e = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.RationalQuadratic(input_dim=1))
        means, variance = e.iterative_train(max_iter=MAXITER)
        np.savez_compressed(
            "/home/stefan/{}_RationalQuadratic".format(i),
            means=means,
            variance=variance,
            signal=signal,
            cps=cps
        )
    
    
