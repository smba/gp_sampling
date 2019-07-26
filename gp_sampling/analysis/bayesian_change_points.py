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
from pygments.unistring import Pc

plt.style.use('bmh')
sns.set_context("paper")

class BayesianChangePointEstimator:
    
    def __init__(self, 
                 signal: np.ndarray,
                 original: np.ndarray,
                 ):
        self.signal = signal
        self.original = original#pd.DataFrame(original).rolling(window=50, center=True).std().values.reshape(1, -1)[0]
        
        self.signal_nonan = signal[~np.isnan(signal)]
        
    def estimate_states(self, n_components = 10) -> dict:
        
        # initialize mixture model
        state_mix = BayesianGaussianMixture(
            n_components, 
            n_init = 1,
            weight_concentration_prior_type = "dirichlet_process",
            verbose = 1,
            tol = 1e-6,
            max_iter = 500
        )
        
        # fit mixture model
        state_mix.fit(self.signal_nonan.reshape(-1, 1))
        
        # get the posterior stuff
        means = state_mix.means_
        sds = state_mix.covariances_
        weights = state_mix.weights_
        
        # count state transitions
        ss = pd.DataFrame(self.signal)
        ss = ss.dropna()
        c = state_mix.predict(ss)
        results = ss
        results["cluster"] = c
        cs = c[np.insert(np.diff(c).astype(np.bool), 0, True)]
            
        clstr = results["cluster"][results.index[0]]
        
        sns.lineplot(np.arange(self.original.shape[0]), self.original, linewidth=0.7, color="black")
        for i, index in enumerate(results.index):
            now = results["cluster"][results.index[i]]
            if clstr != now:
                print("Hier passiert was!", results.index[i-1], index)
                plt.axvline(results.index[i-1], color="darkslateblue")
                plt.axvline(index, color="darkslateblue")
                plt.axvspan(results.index[i-1], index, alpha=0.5, color="darkslateblue")
                #plt.axhline(means[i], index, index)
                clstr = now
        
        return {"mu": means, "sd": sds, "weights": weights, "n_transitions": len(cs) - 1, "n_clusters": len(np.unique(c))}
        
    def change_transform(self):
        
        # do interpolation
        signal_interpolated = pd.DataFrame(self.signal).interpolate().values.ravel().reshape(1, -1)[0]
        signal_interpolated = pd.DataFrame(signal_interpolated).ffill().bfill().values.reshape(1, -1)
        #growth = np.abs(np.diff(signal_interpolated, n=1))
        growth = pd.DataFrame(signal_interpolated).rolling(window=50, center=True).std()
        growth.bfill(inplace=True)
        growth.ffill(inplace=True)
        growth = growth.values.reshape(1, -1)[0]
        growth[np.isnan(growth)] = 0.0
        #growth[growth < 0.2] = 0.0
        growth = growth / np.sum(growth)

        return growth
        
    def change_learning(self, n_components = 15):
        growth = self.change_transform()
        selection = np.random.choice(np.arange(growth.shape[0]), p=growth, size=20000)
        
        # initialize mixture model
        change_mix = BayesianGaussianMixture(
            n_components, 
            n_init = 1,
            weight_concentration_prior_type = "dirichlet_process",
            tol = 1e-9,
            verbose=1,
            max_iter = 1000,
            #init_params = "random"
        )
        change_mix.fit(selection.reshape(-1, 1))
        
        # get the posterior stuff
        means = change_mix.means_
        sds = change_mix.covariances_
        weights = change_mix.weights_
        
        # count state transitions
        ss = pd.DataFrame(self.signal)
        ss = ss.dropna()
        c = change_mix.predict(ss)
        
        return {"mu": means[:,0], "sd": sds, "weights": weights, "n_cps": len(np.unique(c))}
            
    def change_learning_plot(self):
        
        res = self.estimate_states()
        if res["n_clusters"] == 1:
            return
        
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        ax1.set_xlabel("time [revisions]")
        ax1.set_ylabel("performance [s]")
        sns.lineplot(np.arange(self.original.shape[0]), self.original, color="black", ax=ax1, linewidth=0.7)
        sns.scatterplot(np.arange(self.signal.shape[0]), self.signal, marker="X", s=60, color="blue")
        
        
        ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
        ax2.set_xlabel("time [revisions]")
        ax2.set_ylabel("performance change [normalized]")
        growth = self.change_transform()  
        sns.lineplot(np.arange(growth.shape[0]), growth, ax=ax2)
        
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        ax3.set_xlabel("performance [s]")
        ax3.set_ylabel("frequency [normnalized]")
        sns.distplot(self.signal_nonan, ax=ax3, bins=50, label="sample", kde=False, norm_hist=True)
        sns.distplot(self.original[~np.isnan(self.original)], ax=ax3, bins=50, label="ground truth", kde=False, norm_hist=True)
        ax3.legend()
        
        ax4 = plt.subplot2grid((2, 3), (1, 2))
        ax4.set_xlabel("mixture component")
        ax4.set_ylabel("mixture component")
        
        # do the stuff
        res = self.change_learning()
        
        sns.barplot(
            x = np.arange(res["weights"].shape[0]),
            y = res["weights"],
            ax = ax4,
            facecolor="steelblue",
            #palette=mpl.cm.ScalarMappable(cmap='summer').to_rgba(1/res["sd"][:,0,0]) #tau
        )
        
        precision = 1 / res["sd"][:,0,0]
        precision = precision / np.max(precision)
        
        for i in range(res["mu"].shape[0]):            
            ax1.axvline(res["mu"][i], color="firebrick")
        
        plt.show()

if __name__ == "__main__":
    signal = pd.read_csv("/home/stefan/git/gp_sampling/resources/ground_truth/pillow.csv")
    signal = signal[signal.columns[13]][:]
    signal = pd.DataFrame(signal).values.reshape(1, -1)[0]
    plain = np.full(signal.shape, np.nan)
    np.random.seed(66)
    sample = np.random.choice(np.arange(signal.shape[0]), size=50)
    plain[sample] = signal[sample]
    c = BayesianChangePointEstimator(plain, signal)
    #print(c.estimate_states())
    #plt.show()
    c.change_learning_plot()
     
    
    