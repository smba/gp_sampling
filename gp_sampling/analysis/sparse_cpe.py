#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import random

from sklearn.mixture import BayesianGaussianMixture

import numpy as np
import pandas as pd

class BayesianSparseCPE:
    '''
    This class provides two strategies to estimate the location of change-points in a given time-series.
    Along with the estimated location, a measure of uncertainty is given.
    '''

    def __init__(self, observation: np.ndarray, original: np.ndarray = None):
        '''
        Initializes the Change-Point Estimator (CPE) with observed data of a time-series. The length 
        of this array is equal to the total number of the original signal. Un-observed coordinates in the
        time-series are represented by NaN.
        
        Optionally, one can specify the original signal (if available) to compare the inferred change-points 
        with the estimation based on the complete data set.
        
        :param observation: array with observed and un-observed data points
        :param original: array with all measurements of the time-series (optional)
        '''
        
        # observation and original must have the same shape
        assert observation is not None or observation.shape == original.shape
        
        self.observation = observation
        self.original = original
        
    def _interpolate(self, noise: float = 0.0):
        '''
        Fills missing data with a piece-wise linear interpolation.
        
        :param noise: add Gaussian noise to the interpolation with variance $noise (optional)
        '''
        
        # apply linear interpolation
        self.interpolated = pd.DataFrame(self.observation).interpolate().values.ravel().reshape(1, -1)[0]
        self.interpolated = pd.DataFrame(self.interpolated).ffill().bfill().values.reshape(1, -1)[0]
        
        for i, v in enumerate(self.interpolated):
            if np.isnan(self.observation[i]):
                self.interpolated[i] += np.random.normal(0.0, noise)
                
    def _interpolate_variance(self, window = 20, noise: float = 0.0):
        '''
        Calculates the rolling variance for the interpolated time-series.
        
        :param window: window size for the rolling variance
        :param noise: noise level for the interpolation (default is 0.0, cf. _interpolate())
        '''
        
        self._interpolate(noise)
        
        variance = pd.DataFrame(self.interpolated).rolling(window=window, center=True).std()
        variance.bfill(inplace=True)
        variance.ffill(inplace=True)
        variance = variance.values.reshape(1, -1)[0]
        
        return variance / np.sum(variance)
    
    def _interpolate_change(self, diff_n = 1, noise: float = 0.0):
        '''
        Calculates the difference of each revision to the $diff_n previous one based on the interpolation.
        
        :param diff_n: distance to the previous revision (default is 1)
        :param noise: noise level for the interpolation (default is 0.0, cf. _interpolate())
        '''
        self._interpolate(noise)
        
        change = pd.DataFrame(np.abs(np.diff(self.interpolated, n=diff_n)))
        change.bfill(inplace=True)
        change.ffill(inplace=True)
        change = change.values.reshape(1, -1)[0]
        return change / np.sum(change)

    def predict_cp_locations(self, n_components=30, mode = 'variance'):
        '''
        Estimated the variance/change of the interpolation using a truncated Dirichlet Gaussian Mixture Model.
        This automatically infers the most likely number of mixture components.
        
        :param n_components: maximum number of components of the mixture model (default is 30)
        :param mode: interpolation mode, wither 'variance' or 'change' (default is 'variance')
        '''
        
        if mode == 'variance':
            change = self._interpolate_variance()
        elif mode == 'change':
            change = self._interpolate_change()
        
        change = change.reshape(-1, 1)[:,0]
        selection = np.random.choice(np.arange(change.shape[0]), p=change, size = 5 * self.observation.shape[0])
        
        change_mix = BayesianGaussianMixture(
            n_components, 
            n_init = 5,
            tol = 1e-3,
            weight_concentration_prior_type = "dirichlet_process",
            verbose = 1,
            max_iter = 500
        )
        
        change_mix.fit(selection.reshape(-1, 1))
        
        means = change_mix.means_[:,0]
        covariances = change_mix.covariances_[:,0,0]
        
        result = pd.DataFrame({"ev": means, "sd": covariances})
        #result.drop( result[ result['weight'] < 0.01 ].index , inplace=True)
        result.sort_values(by=["mean"], inplace=True)
        result["mean"] = result["mean"].values.astype(int) 
        result.index = np.arange(result.shape[0])

        result = result[result["weight"] > 0.001]
        result.index = np.arange(result.shape[0])

        return result
    
    def predict_cp_interval(self, n_components = 30):
        '''
        Estimates the (phenotypical) levels of observed amplitudes, regardless of order. Consequently, each
        observed time-point is classified. Between each transition from one inferred level to another one, a
        change-point with uniform distirbution is inferred.
        
        :param n_components: maximum number of components of the mixture model (default is 30)
        '''
         
        state_mix = BayesianGaussianMixture(
            n_components, 
            n_init = 5,
            weight_concentration_prior_type = 'dirichlet_process',
            verbose = 1,
            max_iter = 500
        )
        
        observed = self.observation[~np.isnan(self.observation)].reshape(-1, 1)
        
        state_mix.fit(observed)
        
        classified = deepcopy(self.observation)
        predicted = state_mix.predict(classified[~np.isnan(classified)].reshape(-1, 1))
        classified[~np.isnan(classified)] = predicted
        
        last = None
        begin = 0
        for i, c in enumerate(classified):
            if not np.isnan(c):
                last = c
                begin = i
                break
        
        segments = []
        for i in range(begin, classified.shape[0]):
            if not np.isnan(classified[i]):
                if classified[i] != last:
                    s = np.max(np.argwhere(~np.isnan(classified[0:i-1])))
                    segments.append((s, i))
                last = classified[i]
                begin = i
        
        
        # calculate uniform distribution parameters
        result = []
        for segment in segments:
            a = segment[0]
            b = segment[1]
            distro = {
                'ev' : (a + b) / 2,
                'sd': np.sqrt(((b - a + 1)**2 - 1)/12)
            }
            result.append(distro)
        result = pd.DataFrame(result)
        
        return result
        
"""
if __name__ == "__main__":
    signal = pd.read_csv("/home/stefan/git/gp_sampling/resources/ground_truth/pillow.csv")
    signal = signal[signal.columns[11]][:]
    signal = signal.values.reshape(1, -1)[0]
    plain = np.full(signal.shape, np.nan)
    np.random.seed(200)
    
    sample = np.random.choice(np.arange(signal.shape[0]), size=400)
    plain[sample] = signal[sample]
    c = BayesianSparseCPE(plain, signal)
    #c.predict_cp_interval(plot=True, show=True)
    cres = c.predict_cp_locations()
    #cres = c.predict_cp_interval(plot=True, show=True)
    print(cres)
"""
        