#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt

sns.set_style("whitegrid")

def norm_pdf(x, mu, s):
    return (1/(2*np.pi*s)**0.5)*np.e**(-(x-mu)**2/(2*s))

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1-beta)[:-1]])
    return beta * portion_remaining

class BayesianCPE:
    """
    Bayesian changepoint estimatior using truncated Dirichlet mixture models.
    """
    def __init__(self, ys: np.ndarray, n_components: int = 30):
        self.model = pm.Model()
        self.ys = ys
        self._calculate_growth()
        self.n_components = n_components
              
    def _calculate_growth(self, resample = 1000): 
    
        # calculate linear interpolation
        growth = pd.DataFrame(self.ys).interpolate().values.ravel().reshape(1, -1)[0]
        self.growth = np.abs(np.gradient(growth))
            
        # normalize so that the sum is one and smooth it
        self.growth = (self.growth / np.sum(self.growth)) 
        
        # filter step neccesary?
        self.growth = pd.DataFrame(self.growth).rolling(window=10, center=True).mean()
        self.growth.ffill(inplace=True)
        self.growth.bfill(inplace=True)
        self.growth = self.growth.values.reshape(1, -1)[0]
        self.growth = (self.growth / np.sum(self.growth)) 

        # draw samples of likely change point positions
        self.selection = np.random.choice(np.arange(1, self.growth.shape[0]+1), p=self.growth, size=resample)
              
    def generate_priors(self, mode: int = 0):
        
        # maximum number of components the mixture model / truncation point
        K = self.n_components
        
        # number of resampled items
        N = self.selection.shape[0]
        
        with self.model:
            
            # basic model with K normal distribution as mixture components
            if mode == 0:
                # mixture weights are drawn from a Dirichlet distribution
                w = pm.Dirichlet('w', np.ones(K))
                component = pm.Categorical('component', w, shape=N)
                
                # each mixture component has independent mean and concentration
                alpha = pm.Uniform('alpha', 0., 1.0 * N, shape=K)
                beta = pm.HalfCauchy('beta', 1.0, shape=K)
                
                mix = pm.Cauchy('obs', alpha=alpha[component], beta=beta[component], shape=N, observed=self.selection)
                # a = pm.Normal("s", mu=mix, sd=pm.HalfNormal("noiz", sd=10), observed=self.selection, shape=N) 
                
                # preparing MCMC sampling steps
                step1 = pm.Metropolis(vars=[w, alpha, beta, mix])
                step2 = pm.ElemwiseCategorical([component], np.arange(K))
                self.steps = [step1, step2]
                
    def sample_posteriors(self, samples=2500, chains=2, tune=30000, retain=1000):
        with self.model as model:
            try: 
                if retain >= samples:
                    logging.warning("Cannot discard {} samples when only sampling {}; now setting #samples to {}".format(retain, samples, retain+1000))
                    samples = retain + 1000
                    
                #trace = pm.sample(samples, chains=chains, tune=tune)
                trace = pm.sample(samples, self.steps, chains=chains, tune=tune, n_init=1000)
                
                self.trace = dict()
                for rv in ["alpha", "beta", "w"]:
                    self.trace[rv] = np.array( trace[rv][- retain:] )
                
            except ValueError as ver:
                logging.error(ver)
                logging.error(model.check_test_point())
                
    def visualize_model(self):
        mean_centroid = self.trace["alpha"].mean(0)
        tau_centroids = self.trace["beta"].mean(0)
        mean_weights = self.trace["w"].mean(0)
        sigmas = np.array(list(map(lambda t: 1/t, tau_centroids)))
        
        confidence = pd.DataFrame({"x": np.arange(0, self.n_components), "tau":tau_centroids, "w": mean_weights})
        confidence = confidence.sort_values(by=["w"], ascending=False)
        #fig = plt.figure()

        nw = plt.subplot(2, 2, 1)
        nw.title.set_text('Time Series')
        sns.lineplot(np.arange(self.ys.shape[0]), self.ys, linewidth=0.75, color="steelblue", label="observation", ax=nw)
        for i in range(self.n_components):
            alpha = mean_weights[i] / 2
            nw.axvline(x=mean_centroid[i], color="crimson", linewidth=0.9, alpha = alpha)
            nw.axvspan(mean_centroid[i] - sigmas[i], mean_centroid[i] + sigmas[i], alpha=alpha, facecolor="crimson")
        nw.set_xlim((0, self.ys.shape[0]))
        
        ne = plt.subplot(2, 2, 2)
        ne.title.set_text('AVG mixture component weight')
        ne.set_xlabel("average mixture component weight")
        ne.set_ylabel("mixture component")
        
        # see https://stackoverflow.com/questions/39864958/define-hues-by-column-values-in-seaborn-barplot
        sns.barplot(
            data = confidence, 
            x = confidence.index.values, 
            y = "w",  
            ax = ne, 
            palette=mpl.cm.ScalarMappable(cmap='RdYlGn_r').to_rgba(confidence['tau'])
        )
        
        sw = plt.subplot(2, 2, 3)
        sw.title.set_text('Growth')
        sns.lineplot(np.arange(self.growth.shape[0]), self.growth, linewidth=0.8, color="steelblue", label="absolute slope", ax=sw)
        
        se = plt.subplot(2, 2, 4)
        se.title.set_text('Demposition of posterior mixture model')
        xs = np.arange(self.ys.shape[0])
        
            
        posterior = np.zeros(self.ys.shape[0])
        for i in range(self.n_components):
            ys = np.array(list(map(lambda x: norm_pdf(x, mean_centroid[i], sigmas[i]), xs)))
            plt.plot(xs, mean_weights[i]* ys, linewidth=0.5, color="mediumblue", linestyle=":")
            posterior += mean_weights[i] * ys
        sns.lineplot(xs, posterior, linewidth=0.8, color="steelblue")
        
        plt.show()
                             
signal = np.array([20 for i in range(50)] + [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan] + [40 for i in range(50)] +[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan] + [60 for i in range(50)])*1.0
signal += np.array([np.random.normal(0, 0.2) for i in range(signal.shape[0])])

a = BayesianCPE(signal)
a.generate_priors()
a.sample_posteriors()
a.visualize_model()

