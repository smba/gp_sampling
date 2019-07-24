#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper")
plt.style.use('bmh')

def norm_pdf(x, mu, s):
    return (1/(2*np.pi*s)**0.5)*np.e**(-(x-mu)**2/(2*s))

class BayesianStepwiseCPE:
    
    def __init__(self, signal: np.ndarray):
        self.signal = signal
        
    def estimate_states(self, n_components = 10):
        
        K = n_components
        N = self.signal.shape[0]
        
        with pm.Model() as model:
            w = pm.Dirichlet('w', np.ones(K))

            mu = pm.Normal('mu', 0., 10., shape=K)
            tau = pm.Gamma('tau', 1., 1., shape=K)
        
            s = self.signal[~np.isnan(self.signal)]
            x_obs = pm.NormalMixture('x_obs', w, mu, tau=tau, observed=s)
            
            self.trace = dict()
            trace = pm.sample(5000, n_init=10000, tune=1000, random_seed=123)
            retain = 500
            for rv in ["mu", "tau", "w"]:
                self.trace[rv] = np.array( trace[rv][- retain:] )
                
        ax1 = plt.subplot(2,2,1)
        #sns.distplot(self.signal[~np.isnan(self.signal)], kde=False, ax=ax1, bins=100, norm_hist=True)
        
        mean_centroid = self.trace["mu"].mean(0)
        tau_centroids = self.trace["tau"].mean(0)
        mean_weights = self.trace["w"].mean(0)
        sigmas = np.array(list(map(lambda t: 1/t, tau_centroids)))
        xs = np.linspace(min(self.signal), max(self.signal), 200)
        
        posterior = np.zeros(xs.shape[0])
        for i in range(n_components):
            print( (mean_centroid[i], mean_weights[i], tau_centroids[i]) )
            ys = np.array(list(map(lambda x: norm_pdf(x, mean_centroid[i], sigmas[i]), xs)))
            sns.lineplot(xs, mean_weights[i]* ys, linewidth=0.5, color="mediumblue", linestyle=":", ax=ax1)
            posterior += mean_weights[i] * ys
        sns.lineplot(xs, posterior, linewidth=1.2, color="steelblue", ax=ax1, label="GMM")
        
        
        plt.show()
                            
if __name__ == "__main__":
    a = [np.random.normal(10, 0.4) for i in range(100)] + [np.random.normal(6, 0.5) for i in range(50)] + [np.random.normal(14, 0.3) for i in range(50)]
    a = np.array(a)
    sample = np.random.choice(np.arange(a.shape[0]), size = 50)
    signal = np.full(a.shape, np.nan)
    signal[sample] = a[sample]
    
    c = BayesianStepwiseCPE(a)
    c.estimate_states(10)
                    
                    