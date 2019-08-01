#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import logging
import random

from scipy.stats.mstats_basic import mannwhitneyu

import metrics
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler


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
        
        logging.warn("Predicting CP locations")
        
        if mode == 'variance':
            change = self._interpolate_variance()
        elif mode == 'change':
            change = self._interpolate_change()
        
        change = change.reshape(-1, 1)[:,0]
        selection = np.random.choice(np.arange(change.shape[0]), p=change, size = 5 * self.observation.shape[0])
        
        change_mix = BayesianGaussianMixture(
            n_components, 
            n_init = 1,
            weight_concentration_prior_type = "dirichlet_process",
            verbose = 1,
            max_iter = 500
        )
        
        change_mix.fit(selection.reshape(-1, 1))
        
        weights = change_mix.weights_[:]
        means = change_mix.means_[:,0]
        covariances = change_mix.covariances_[:,0,0]
        
        result = pd.DataFrame({"ev": means, "sd": covariances, "weight": weights})
        result.sort_values(by=["sd"], inplace=True)
        result["ev"] = result["ev"].values.astype(int) 
        result.index = np.arange(result.shape[0])
        result = result[result["weight"] > 0.001]
        result.index = np.arange(result.shape[0])
        
        # measure the significance and effect size
        #significances = []
        #a12s = []
        W = 30
        for i in range(result.shape[0]):
            ev = result["ev"][i]
            left = 0 if i == 0 else result["ev"][i-1]
            right = self.interpolated.shape[0] if i == result.shape[0]-1 else result["ev"][i+1]
            segment1 = self.interpolated[left : ev]
            segment2 = self.interpolated[ev: right]
            print(len(segment1), len(segment2))
            #significances.append(mannwhitneyu(segment1, segment2).pvalue)
            #a12s.append(2*np.abs(metrics.a12(segment1, segment2) - 0.5))
        
        #result["significance"] = significances
        #result["a12"] = a12s
        #print(result)
        #result = result[result["significance"] <= 0.05]
        #result = result[result["a12"] > 0.5]
        result.index = np.arange(result.shape[0])
        return result
    
    def predict_cp_interval(self, n_components = 30):
        '''
        Estimates the (phenotypical) levels of observed amplitudes, regardless of order. Consequently, each
        observed time-point is classified. Between each transition from one inferred level to another one, a
        change-point with uniform distirbution is inferred.
        
        :param n_components: maximum number of components of the mixture model (default is 30)
        '''
        
        logging.warn("Predicting CP intervals")
         
        state_mix = BayesianGaussianMixture(
            n_components, 
            n_init = 10,
            weight_concentration_prior_type = 'dirichlet_distribution',
            verbose = 1,
            max_iter = 500,
            tol=1e-12
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
                'begin': a,
                'end': b
            }
            result.append(distro)
        result = pd.DataFrame(result)
        
        return result, state_mix
        

if __name__ == "__main__":
    import analysis.change_points as cp
    import matplotlib.pyplot as plt
    import seaborn as sns
    import ruptures
    
    signal = pd.read_csv("/home/stefan/git/gp_sampling/resources/ground_truth/pillow.csv")
    signal = signal[signal.columns[11]][:]
    signal = signal.values.reshape(1, -1)[0]
    plain = np.full(signal.shape, np.nan)
    np.random.seed(350)
    
    sample = np.random.choice(np.arange(signal.shape[0]), size=450)
    plain[sample] = signal[sample]
    c = BayesianSparseCPE(plain, signal)    
    intervals, state_mix = c.predict_cp_interval(30)
    locations = c.predict_cp_locations(mode="change")
    
    print(locations)
    
    sns.lineplot(np.arange(signal.shape[0]), signal, linewidth=0.3, color="black")
    #plt.scatter(np.arange(plain.shape[0]), plain, marker="X")
    
    for i in range(intervals.shape[0]):
        plt.axvspan(intervals["begin"][i], intervals["end"][i], alpha=0.5, color="moccasin")
    
    for i in range(state_mix.means_.shape[0]):
        if state_mix.weights_[i] > 0.001:
            plt.axhline(state_mix.means_[i], color="silver", linestyle=":", linewidth=0.8)
    # = cres[cres["a12"] > 0.75]
    #cres.index = np.arange(cres.shape[0])
    #for i in range(cres.shape[0]):
    #    plt.axvline(cres["ev"][i], color="lime")
    plt.show()
        