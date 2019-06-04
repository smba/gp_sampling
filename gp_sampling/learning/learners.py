from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import gpflow
import logging
import numpy as np
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
                 init_training: int = 3):
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
        
    def _train(self) -> None:
        '''
        
        '''
        xs = self.xs[self.training_set] * 1.0
        self.model = gpflow.models.GPR(
                xs, self.ys[self.training_set],
                kern = self.kernel
        )
        self.model.likelihood.variance=0.01
        gpflow.train.ScipyOptimizer().minimize(self.model)

    
    def iterative_train(self, max_iter: int = 200):
        '''
        Train, acquire next, and repeat. This is the main training method for the IterativeLearner.
        
        :param max_iter: Maximum number of iterations, i.e., observations to include in the training set.
        '''
        self._train()
        counter = 0
        while not counter > max_iter:
            self.acquire_next()
            self._train()
            counter += 1
        
    @abstractmethod
    def acquire_next(self) -> None:
        ...
        
    def predict(self) -> Tuple[float, float]:
        '''
        Calculate an estimate and return mean and variance
        
        @return:  tuple (mean, variance) of the GP estimate
        '''
        means, variance = self.model.predict_y(self.xs)
        means = self._transform_inverse(means)
        variance = np.abs(self._transform_inverse(variance))
           
        return means, variance
    
    def _status(self):
        print("Training set: {} elements".format(len(set(self.training_set))))
       
class IterativeRandomLearner(IterativeLearner):
    '''
    The IterativeRandomLearner adds random sampling to the IterativeLearner by implementing 
    the acquire_next method accordingly. The acquisition function selects an arbitrary 
    data point and adds this to the training set.
    '''
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kernel: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 3):
        IterativeLearner.__init__(self, xs, ys, kernel, init_training)
                  
    def acquire_next(self) -> None:
        '''
        This acquisition function adds an arbitrary data point to the training set.
        '''
        not_training = set(self.xs.reshape(1, -1)[0]).difference(set(self.training_set))
        not_training = np.array(not_training)
        nxt = [np.random.choice(not_training)]
        self.training_set = np.append(self.training_set, nxt)
        
class ActiveLearner(IterativeLearner):
    '''
    The ActiveLearner adds active uncertainty-guided sampling to the IterativeLearner by
    implementing the acquire_next method accordingly. The acquisition function selects the 
    data point with the largest uncertainty/smallest confidence and adds this to the training set.
    '''
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kernel: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 3):
        IterativeLearner.__init__(self, xs, ys, kernel, init_training)
                  
    def acquire_next(self) -> None:
        '''
        This acquisition function adds the data point with the highest uncertainty to the training set.
        '''
        std = self.predict()[1]
        not_training = set(self.xs.reshape(1, -1)[0]).difference(set(self.training_set))
        not_training = np.array(not_training)
        std[self.training_set] = 0.0
        nxt = [np.argmax(std)]
        self.training_set = np.append(self.training_set, nxt)
   
class BalancedActiveLearner(ActiveLearner):
    '''
    The BalancedActiveLearner, unlike the ActiveLearner does not only query new data points to explore, 
    but excludes data points from the training set if the training set size exceeds a balance limit. This 
    is to limit the training effort and extract data points contributing best to the GP estimate.
    '''
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray, 
                 kernel: gpflow.kernels.Kernel = gpflow.kernels.RBF(input_dim=1),
                 init_training: int = 3,
                 balance_limit = 200):
        ActiveLearner.__init__(self, xs=xs, ys=ys, kernel=kernel, init_training=init_training)
        self.balance_limit=balance_limit
        
    def acquire_next(self) -> None:
        '''
        This acquisition function adds the data point with the largest uncertainty to the training set. In addition, 
        if the size of the training set exceeds the limit specified upon instantiation, also, the data point with
        the least uncertainty (i.e. highest confidence) from the training set is excluded.
        
        This is according to the original implementation of Roberts et al. (2012)
        
        '''
        std = self.predict()[1]
        
        to_remove = pd.DataFrame(std).iloc[self.training_set]
        to_remove = to_remove.idxmin().iloc[0]
        
        not_training = set(self.xs.reshape(1, -1)[0]).difference(set(self.training_set))
        not_training = np.array(not_training)
        std[self.training_set] = 0.0
        nxt = [np.argmax(std)]
        
        if len(self.training_set) == self.balance_limit:
            self.training_set = np.delete(self.training_set, np.where(self.training_set == to_remove))
        self.training_set = np.append(self.training_set, nxt)

            
        
