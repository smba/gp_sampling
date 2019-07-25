#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture

sns.set_context("paper")
plt.style.use('bmh')

def norm_pdf(x, mu, s):
    return (1/(2*np.pi*s)**0.5)*np.e**(-(x-mu)**2/(2*s))

class BayesianStepwiseCPE:
    
    def __init__(self, signal: np.ndarray):
        self.signal = signal
        
    def estimate_states(self, n_components = 15):
        mix = BayesianGaussianMixture(
            n_components, 
            n_init = 25,
            weight_concentration_prior_type = "dirichlet_process",
            #warm_start = True,
            verbose = 1,
            tol = 1e-12,
            max_iter = 1000
        )
        s = self.signal[~np.isnan(self.signal)].reshape(-1, 1)
        mix.fit(s)
        
        a = plt.subplot(1,2,1)
        
        #sns.distplot(s, ax = a, bins=50, color="darkgrey", norm_hist=True)
        
        means = mix.means_
        sds = mix.covariances_
        weights = mix.weights_
        rand = 0.1
        xmin = np.min(s) - rand * np.mean(s)
        xmax = np.max(s) + rand * np.mean(s)
        xs = np.linspace(xmin, xmax, 500)
        
        for i in range(means.shape[0]):
            sd = sds[i, 0, 0]
            mu = means[i ,0]
            
            if weights[i] > 0.05:
                pdf = lambda x: norm.pdf(x, loc = mu, scale = sd)
                ys = list(map(pdf, xs))
    
                sns.lineplot(xs, ys, color="firebrick", ax=a, alpha=weights[i]*2)
        
        b = plt.subplot(1,2,2)
        sns.barplot(list(range(n_components)), weights, ax = b, facecolor="steelblue")
        
        cps_cnt = 0
        ss = pd.DataFrame(self.signal)
        ss = ss.dropna()
        c = mix.predict(ss)
        clusters = np.unique(c)
        cs = c[np.insert(np.diff(c).astype(np.bool), 0, True)]
        print("identified {} states and {} transitions".format(len(clusters), len(cs) - 1))
                
        plt.show()
        
                            
if __name__ == "__main__":
    signal = pd.read_csv("/home/stefan/git/gp_sampling/resources/ground_truth/pillow.csv")
    signal = signal[signal.columns[23]][:]
    signal = pd.DataFrame(signal).values.reshape(1, -1)[0]
    plain = np.full(signal.shape, np.nan)
    np.random.seed(42)
    sample = np.random.choice(np.arange(signal.shape[0]), size=7999)
    plain[sample] = signal[sample]
    
    c = BayesianStepwiseCPE(plain)
    c.estimate_states()
                    
                    