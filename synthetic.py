import os
import random 
import sys
from gp_sampling.learning.kernels import Brownian
from gp_sampling.learning.learners import ActiveLearner
import numpy as np
import gpflow
import uuid

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
    u = uuid.uuid4()    
    i = 42
    signal, cps = create_signal()
    prefix = sys.argv[1]
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    
    MAXITER = 50
    
    a = ActiveLearner(np.arange(0, 1000), signal, Brownian())
    means, variance = a.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "{}{}_{}_Brownian".format(prefix, i, u),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    b = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.Matern32(input_dim=1))
    means, variance = b.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "{}{}_{}_Matern32".format(prefix, i, u),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    c = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.Matern52(input_dim=1))
    means, variance = c.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "{}{}_{}_Matern52".format(prefix, i, u),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    d = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.RBF(input_dim=1))
    means, variance = d.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "{}{}_{}_RBF".format(prefix, i, u),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    e = ActiveLearner(np.arange(0, 1000), signal, gpflow.kernels.RationalQuadratic(input_dim=1))
    means, variance = e.iterative_train(max_iter=MAXITER)
    np.savez_compressed(
        "{}{}_{}_RationalQuadratic".format(prefix, i, u),
        means=means,
        variance=variance,
        signal=signal,
        cps=cps
    )
    
    
