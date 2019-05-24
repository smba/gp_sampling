from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import gpflow

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import scipy.signal as sig
import sklearn.preprocessing as preprocessing

class IterativeLearner(ABC):
    '''
    An iterative learner pursues a cycle of training, validation, and data acquisition. Different variants can be are
    specified with different GP regression base implementations and different acquisition functions.
    
    @param xs: input values
    @param ys: output values (ground truth observations)
    @param kern: covariance function/kernel for the iterative learning. Must be instance of gpflow.kernels.Kernel     
    @param init_training: size of the initial training set (10 is default)
    '''
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kern: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 10):
        self.kernel = kern
        self.scaler = preprocessing.StandardScaler().fit(ys.reshape(-1, 1))
        self.xs = xs.reshape(-1, 1)
        self.ys = self.scaler.transform(ys.reshape(-1, 1))
        self.training_set = np.random.choice(self.xs.reshape(1, -1)[0], size=init_training)
        self.training_set = np.append(self.training_set, [0, len(self.training_set)])
        
    
    def _transform(self, ys: np.ndarray) -> np.ndarray:
        '''
        Transform an observation to internal scales.
        
        :param ys: observations to transform
        :return: transformed observations (internal scale)
        '''
        return self.scaler.transform(ys)
    
    def _transform_inverse(self, ys: np.ndarray) -> np.ndarray:
        '''
        Transform an observation back from internal scales back to its original form.
        
        :param ys: internal values to transform
        :return: observations (normal scale)
        '''
        return self.scaler.inverse_transform(ys)
        
    def train(self) -> None:
        xs = self.xs[self.training_set] * 1.0
        self.model = gpflow.models.GPR(
                xs, self.ys[self.training_set],
                kern = self.kernel
        )
        self.model.likelihood.variance=0.01
        gpflow.train.ScipyOptimizer().minimize(self.model)

    
    def iterative_train(self, max_iter: int = 200):
        self.train()
        counter = 0
        while not counter > max_iter:
            self.acquire_next()
            self.train()
            counter += 1
        
    @abstractmethod
    def acquire_next(self) -> None:
        ...
        
    def predict(self) -> Tuple[float, float]:
        means, variance = self.model.predict_y(self.xs)
        means = self._transform_inverse(means)
        variance = self._transform_inverse(variance)      
        return means, variance
        
    def change_point_estimation(self, cps: Sequence[int], n = 5):
        mean, variance = self.predict()
        
        algo = rpt.Window(width=30, model="l2").fit(mean )
        cp_estimation = algo.predict(pen=10)

        tps = []
        fps = []
        fng = []
        for ce in cp_estimation:
            matching = [c in range(ce - n, ce + n) for c in cps]
            if any(matching):
                tps.append(ce)
            else:
                fps.append(ce)
        
        for cp in cps:
            matching = [cp in range(ce - n, ce + n) for ce in cp_estimation]
            if not any(matching):
                fng.append(cp)
        
        if (len(tps) + len(fps)) == 0:
            precision = 1.0
        else:
            precision = len(tps)/(len(tps) + len(fps))
        
        if (len(tps) + len(fng)) == 0:
            recall = 1.0
        else:
            recall = len(tps)/(len(tps) + len(fng))
        
        
        
        ys = self._transform_inverse(self.ys)
        
        fig, ax1 = plt.subplots(figsize=(13,6))
        ax1.plot(self.xs, ys, color="black", label="ground truth", linewidth=0.8)
        ax1.plot(self.xs, mean, label="GP estimate", color="blue")
        #plt.plot(self.xs, pd.DataFrame(mean).rolling(window=30, center =True).median(), label="GP estimation (median)", color="dodgerblue")
        mean = mean.reshape(1, -1)[0]
        variance = variance.reshape(1, -1)[0]
        ax1.fill_between(self.xs.reshape(1, -1)[0], mean - variance, mean + variance, alpha=0.25, color="blue", label="GP estimate uncertainty")
        #plt.scatter(self.training_set, ys[self.training_set], marker=".", s=120, color="blue", label="observations")
        ax1.set_ylabel("performance")
        ax1.scatter(self.training_set, ys[self.training_set], color="blue")
        plt.xlabel("time (revisions)")
        
        ax1.scatter(cps, self._transform_inverse(self.ys[cps]), label="ground truth CPs", color="black", marker="x")
        for cp in cp_estimation:
            ax1.axvspan(cp - n, cp + n, alpha=0.5, color='dodgerblue')

        diff = np.diff(mean, n=1)
        #print(diff)
        ht = 0.045
        peaks = sig.find_peaks(np.abs(diff), height=ht)[0]
        #print(np.min(diff), np.max(diff))
        ax1.legend()
        #print(mape(ys.reshape(1, -1)[0], mean))
        plt.show()
        
class IterativeRandomLearner(IterativeLearner):
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kernel: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 10):
        super.__init__(self, xs, ys, kernel, init_training)
                  
    def acquire_next(self) -> None:
        not_training = set(self.xs.reshape(1, -1)[0]).difference(set(self.training_set))
        not_training = np.array(not_training)
        next = [np.random.choice(not_training)]
        self.training_set = np.append(self.training_set, next)
        
class ActiveLearner(IterativeLearner):
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kernel: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 10):
        super.__init__(self, xs, ys, kernel, init_training)
                  
    def acquire_next(self) -> None:
        mean, std = self.predict()
        not_training = set(self.xs.reshape(1, -1)[0]).difference(set(self.training_set))
        not_training = np.array(not_training)
        std[self.training_set] = 0.0
        nxt = [np.argmax(std)]
        self.training_set = np.append(self.training_set, nxt)
   
class ActiveBalancedLearner(IterativeLearner):
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kernel: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 10):
        IterativeLearner.__init__(self, xs=xs, ys=ys, kern=kernel, init_training=init_training)
        self.balance_limit=200
        
    def acquire_next(self) -> None:
        mean, std = self.predict()
        not_training = set(self.xs.reshape(1, -1)[0]).difference(set(self.training_set))
        not_training = np.array(not_training)
        
        std_max = std
        std_min = std
        
        std_max[self.training_set] = 0
        nxt = np.argmax(std_max)
        print(nxt)
        
        std_min[self.training_set] = np.iinfo(np.int64(10)).max
        to_remove = np.argmin(std_min)
        
        # balancing the training set
        if self.training_set.shape[0] + 1 < self.balance_limit:
            self.training_set = np.append(self.training_set, [nxt])
        else:
            loc = np.where(self.training_set, to_remove)
            self.training_set = np.delete(self.training_set, loc)
            self.training_set = np.append(self.training_set, [nxt])
        
        print(self.training_set)
        self.change_point_estimation(cps=[])