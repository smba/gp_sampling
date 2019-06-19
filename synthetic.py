import os
import random 
import sys

import gpflow

from gp_sampling.learning.kernels import Brownian
from gp_sampling.learning.learners import ActiveLearner
import learning.kernels as kernels
import numpy as np


def create_signal():
    cps = np.random.choice(list(range(5)), size=random.randint(1, 3))
    sig = [0]
    for i in range(999):
        s = sig[-1]
        if i in cps:
            s += np.random.normal(0, 3)
        s += 0.3 * np.random.normal(0, 1)
        sig.append(s)
    return np.array(sig), cps


if __name__ == "__main__":
    i = sys.argv[1]
    signal, cps = create_signal()
    
    MAXITER = 5
    
    a = ActiveLearner(np.arange(0, 50), signal, Brownian())
    means, variance = a.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "/home/stefan/{}_Brownian".format(i),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    b = ActiveLearner(np.arange(0, 50), signal, gpflow.kernels.Matern32(input_dim=1))
    means, variance = b.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "/home/stefan/{}_Matern32".format(i),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    c = ActiveLearner(np.arange(0, 50), signal, gpflow.kernels.Matern52(input_dim=1))
    means, variance = c.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "/home/stefan/{}_Matern52".format(i),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    d = ActiveLearner(np.arange(0, 50), signal, gpflow.kernels.RBF(input_dim=1))
    means, variance = d.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "/home/stefan/{}_RBF".format(i),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    e = ActiveLearner(np.arange(0, 50), signal, gpflow.kernels.RationalQuadratic(input_dim=1))
    means, variance = e.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "/home/stefan/{}_RationalQuadratic".format(i),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    
