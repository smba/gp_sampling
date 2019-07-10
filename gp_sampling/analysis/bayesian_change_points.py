#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt


sns.set_style("darkgrid")

def norm_pdf(x, mu, s):
    return (1/(2*np.pi*s)**0.5)*np.e**(-(x-mu)**2/(2*s))

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1-beta)[:-1]])
    return beta * portion_remaining

class BayesianCPE:
    """
    
    """
    def __init__(self, ys: np.ndarray, n_components = 15):
        """
        """
        self.n_components = n_components
        self.model = pm.Model()
        self.ys = ys
        self._calculate_slopes()
              
    def _calculate_slopes(self, n = 1000):  
        """
        """      
        yz = pd.DataFrame(self.ys).interpolate().values.ravel().reshape(1, -1)[0]
        self.slopes = np.abs(np.gradient(yz))
              
        self.slopes = (self.slopes / np.sum(self.slopes))
        self.selection = np.random.choice(np.arange(1, self.slopes.shape[0]+1), p=self.slopes, size=n)
              
    def generate_priors(self):
        """
        """
        with self.model:
            w = pm.Dirichlet('w', np.ones(self.n_components))      
                 
            mu = pm.DiscreteUniform("mu", 0, self.ys.shape[0], shape=self.n_components)
            tau = pm.Gamma('tau', 1, 1., shape=self.n_components)
            
            components = pm.Normal.dist(mu=mu, tau=tau, shape=self.n_components)
            obs = pm.Mixture('obs', w=w, comp_dists = components, observed=self.selection)
      
    def sample_posteriors(self):
        """
        """
        with self.model as model:
            try:
                self.trace = pm.sample(100, n_init=1000, tune=100)
            except ValueError as ver:
                logging.error(ver)
                logging.error(model.check_test_point())
                
    def visualize_model(self):
        
        #print(self.trace["obs"])
        
        mean_centroid = self.trace["mu"][-1000:].mean(0)
        tau_centroids = self.trace["tau"][-1000:].mean(0)
        mean_weights = self.trace["w"][-1000:].mean(0)
        
        fig = plt.figure()

        nw =plt.subplot(2, 2, 1)
        nw.title.set_text('Time Series')
        sns.lineplot(np.arange(self.ys.shape[0]), self.ys, linewidth=0.75, color="black", label="observation", ax=nw)
        for i in range(self.n_components):
            nw.axvline(x=mean_centroid[i], color="blue", linewidth=0.9, alpha= min(4*mean_weights[i], 1))
        
        ne = plt.subplot(2, 2, 2)
        ne.title.set_text('Average mixture component weight')
        ne.set_xlabel("component")
        ne.set_ylabel("avg. mixture component weight")
        sns.barplot(np.arange(mean_weights.shape[0]), mean_weights, color="blue", ax=ne, alpha=0.5)
        
        sw = plt.subplot(2, 2, 3)
        sw.title.set_text('Absolute Slopes')
        sns.lineplot(np.arange(self.slopes.shape[0]), self.slopes, linewidth=0.8, color="black", label="absolute slope", ax=sw)
        
        se = plt.subplot(2, 2, 4)
        se.title.set_text('Composition of posterior mixture model')
        xs = np.arange(self.ys.shape[0])
        sigmas = np.array(list(map(lambda t: 1/t, tau_centroids)))
        
        posterior = np.zeros(self.ys.shape[0])
        for i in range(self.n_components):
            ys = np.array(list(map(lambda x: norm_pdf(x, mean_centroid[i], sigmas[i]), xs)))
            sns.lineplot(xs, mean_weights[i]* ys, linewidth=0.5, color="cornflowerblue")
            posterior += mean_weights[i] * ys
        sns.lineplot(xs, posterior, linewidth=0.8, color="mediumblue")
        
        plt.show()
                             
signal = np.array([20 for i in range(50)] + [np.nan,np.nan,np.nan,np.nan] + [40 for i in range(50)] + [np.nan,np.nan,np.nan,np.nan] +  [60 for i in range(50)])*1.0
signal += np.array([np.random.normal(0, 0.3) for i in range(signal.shape[0])])

a = BayesianCPE(signal)
a.generate_priors()
a.sample_posteriors()
a.visualize_model()

