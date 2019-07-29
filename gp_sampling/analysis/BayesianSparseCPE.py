#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from conda.common import signals
from scipy.stats import norm, mannwhitneyu
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LinearRegression
import logging
from youtube_dl.aes import mix_column
from oauthlib.uri_validate import ls32

#plt.style.use('bmh')
sns.set_context("paper")

class BayesianSparseCPE(object):
    '''
    classdocs
    '''

    def __init__(
            self, 
            observation: np.ndarray,
            original: np.ndarray = None
        ):
        
        assert observation is not None or observation.shape == original.shape
        
        self.observation = observation
        self.original = original
        
    def _interpolate(self, noise = 0.00):
        
        # apply linear interpolation
        self.interpolated = pd.DataFrame(self.observation).interpolate().values.ravel().reshape(1, -1)[0]
        self.interpolated = pd.DataFrame(self.interpolated).ffill().bfill().values.reshape(1, -1)[0]
        
        for i, v in enumerate(self.interpolated):
            if np.isnan(self.observation[i]):
                self.interpolated[i] += np.random.normal(0.0, noise)
                
    def _interpolate_variance(self, window = 20):
        self._interpolate()
        
        variance = pd.DataFrame(self.interpolated).rolling(window=window, center=True).std()
        variance.bfill(inplace=True)
        variance.ffill(inplace=True)
        variance = variance.values.reshape(1, -1)[0]
        return variance / np.sum(variance)
    
    def _interpolate_change(self, diff_n = 1):
        self._interpolate()
        
        change = pd.DataFrame(np.abs(np.diff(self.interpolated, n=diff_n)))
        change.bfill(inplace=True)
        change.ffill(inplace=True)
        change = change.values.reshape(1, -1)[0]
        return change / np.sum(change)

    def predict_cp_locations(self, mode = 'variance'):
        pass

    def predict_cp_interval(self, plot=False, show=False, N = 15):
        change = self._interpolate_variance().reshape(-1, 1)[:,0]
        selection = np.random.choice(np.arange(change.shape[0]), p=change, size=10000)
        # initialize mixture model100
        change_mix = BayesianGaussianMixture(
            N, 
            n_init = 1,
            weight_concentration_prior_type = "dirichlet_process",
            verbose = 1,
            max_iter = 500
        )
        
        change_mix.fit(selection.reshape(-1, 1))
        
    
        if plot:
            ax1 = plt.subplot(2, 1, 1)
            
            if self.original is not None:
                sns.lineplot(np.arange(self.original.shape[0]), self.original, ax=ax1, color="grey", alpha=0.5)
            sns.lineplot(np.arange(self.interpolated.shape[0]), self.interpolated, color="black", ax=ax1)
            sns.scatterplot(np.arange(self.observation.shape[0]), self.observation, marker="X", color="crimson", ax=ax1, s=100)
        
            for i in range(change_mix.weights_.shape[0]):
                plt.axvline(change_mix.weights_[i], color="steelblue")
        
            ax2 = plt.subplot(2, 1, 2)
            sns.barplot(np.arange(change_mix.weights_.shape[0]), change_mix.weights_, color="firebrick")
            
        if show:
            plt.show()
        else:
            plt.draw()
            plt.savefig("out.pdf", bbox_inches="tight")
    
    """
    def _cluster_means(self):
        
        # drop NaN
        observation = self.observation[~np.isnan(self.observation)]
        
        # initialize mixture model
        state_mix = BayesianGaussianMixture(
            30, 
            n_init = 1,
            weight_concentration_prior_type = "dirichlet_process",
            verbose = 1,
            max_iter = 500
        )
        
        # fit mixture model
        state_mix.fit(observation.reshape(-1, 1))
        
        # get the posterior stuff
        means = state_mix.means_
        sds = state_mix.covariances_
        weights = state_mix.weights_
        
        return means, sds, weights
    """
    
if __name__ == "__main__":
    signal = pd.read_csv("/home/stefan/git/gp_sampling/resources/ground_truth/pillow.csv")
    signal = signal[signal.columns[11]][:]
    signal = pd.DataFrame(signal).values.reshape(1, -1)[0]
    plain = np.full(signal.shape, np.nan)
    np.random.seed(200)
    sample = np.random.choice(np.arange(signal.shape[0]), size=400)
    plain[sample] = signal[sample]
        
    c = BayesianSparseCPE(plain, signal)
    c.predict_cp_interval(plot=True, show=True)#(0.05)10
        