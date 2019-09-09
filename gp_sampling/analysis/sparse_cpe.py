#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import random

from sklearn.mixture import BayesianGaussianMixture,GaussianMixture

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
import xmltodict


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
            max_iter = 100
        )
        
        change_mix.fit(selection.reshape(-1, 1))
        
        means = change_mix.means_[:,0]
        covariances = change_mix.covariances_[:,0,0]
        weights = change_mix.weights_
        result = pd.DataFrame({"ev": means, "sd": covariances, "w": weights})
        result.drop( result[ result['w'] < 0.05 ].index , inplace=True)
        result.sort_values(by=["ev"], inplace=True)
        result["ev"] = result["ev"].values.astype(int) 
        result.index = np.arange(result.shape[0])

        #result = result[result["weights"] > 0.001]
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
    
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

class BisectionSparseCPE:
    def __init__(self, signal: np.ndarray):
        self.signal = signal
    
    def benery_segmentation(self, threshold=5, init = 1):
    
        signal = self.signal
        sample = [0, (signal.shape[0] - 1) // 2, signal.shape[0] - 1]
        changes = np.array([])
        
        con = True
        
        while con:
            changes = np.array([])
            segments = list(zip(sample[:-1], sample[1:]))
            
            for start, end in segments:
                if np.abs(start - end) <= 1:
                    change = 0
                else:
                    change = np.abs(signal[start] - signal[end])
                changes = np.append(changes, change)
                
            nxt = np.sum(segments[np.argmax(changes)]) // 2
            sample = sorted(sample + [nxt])
            con = any(changes > threshold)
            
        cps = []
        for i in range(len(segments)):
            if changes[i] == 0 and np.abs(signal[segments[i][0]] - signal[segments[i][1]]) > threshold:
                cps.append(segments[i][1])
        return cps, sample
    

def parseConfiurations(path_to_vm, path_to_configs):
    decipher = lambda s: (s, 1) if not ';' in s else tuple( s.split(';') )

    configuration_vectors = []

    with open(path_to_vm) as xml_file:
        doc = xmltodict.parse(xml_file.read())
        binary_xml = doc['vm']['binaryOptions']['configurationOption']
        binary_features = []
        for bfeature in binary_xml:
            binary_features.append( bfeature['name'] )

        numeric_xml = doc['vm']['numericOptions']['configurationOption']
        numeric_features = []
        for nfeature in numeric_xml:
            numeric_features.append( nfeature['name'] )

        features = binary_features[1:] + numeric_features
        
        with open(path_to_configs) as f:
            lines = f.readlines()

            for line in lines:
                line = line[line.find('"') + 1:]
                line = line[:line.find('"')]
                line = filter(lambda x: x != '', line.split('%;%'))
                line = dict( map(decipher, line) )

                fline = []
                fdict = {}
                for feature in features:
                    if feature not in line.keys():
                        fline.append(0)

                        fdict[feature] = 0

                    else:
                        fline.append( line[feature] )

                        fdict[feature] = line[feature]
                configuration_vectors.append( fdict )
    return configuration_vectors

if __name__ == "__main__":
    signal = pd.read_csv("/home/stefan/git/gp_sampling/resources/ground_truth/lrzip.csv")
    vm = "/home/stefan/git/gp_sampling/resources/ground_truth/lrzip.xml"
    cfgs = "/home/stefan/git/gp_sampling/resources/ground_truth/lrzip.config"
    cfgs = parseConfiurations(vm, cfgs)
    
    cfgs = pd.DataFrame(cfgs)
    cfgs = cfgs[['Encryption', 'LzoCompressibilityTesting', 'UnlimitedWindow','Compression', 'Bzip2', 'Gzip', 'Lzo', 'Zpaq']]
    
    signal = signal[signal.columns[1:]]
    
    
    signal -= signal.min()
    signal /= signal.max()
    signal.ffill(inplace=True)
    signal.bfill(inplace=True)
    signal = signal
    #signal = signal.T
    
    # make interpolations
    observation = np.full((cfgs.shape[0], signal.shape[0]), np.nan)
    
    sampling_rate = 0.15
    #signalt = signal.T
    N = int(sampling_rate * signal.shape[0])
    for i in range(observation.shape[0]-1): # per config
        sample = np.random.choice(np.arange(signal.shape[0]), size=N)
        observation[i,sample] = signal[str(i)][sample].values
    
    interpolated = pd.DataFrame(observation).interpolate(method="akima", axis=1, order=5)
    interpolated.ffill(inplace=True)#
    interpolated.bfill(inplace=True)
    W = 9
    variance = pd.DataFrame(interpolated).rolling(window=W, center=True, axis=1).std()#**2
    variance = variance.T
    print(variance.shape)
    # create clusterizable data frame
    new = []
    for i, c in enumerate(cfgs.values[:]):
        for t in range(variance.shape[0]-2):
            vector = np.append(c, [t, variance[i].values[t]])
            new.append(vector)
    new = pd.DataFrame(new)
    new.columns = np.append(cfgs.columns, ["time", "variance"])
    news = new.values
    news[:, -1] *= 1 #kosmetik
    
    components= 25
    mm = BayesianGaussianMixture(
            components, 
            covariance_type="tied",
            #init_params="random",
            n_init = 5,
            tol = 1e-3,
            weight_concentration_prior_type = "dirichlet_distribution",
            verbose = 3,
            max_iter = 100,
        )
        
    #mm.fit(news[:,:-1], news[:,-1])
    
    #for i in range(components):
    #    print(mm.weights_[i], mm.means_[i][-1])
        
    #plt.pcolormesh(variance.T*100, cmap="viridis")
    #plt.colorbar()
    
    #print(variance.shape)
    variance = variance.values
    plt.bar(np.arange(signal.shape[0]), np.sum(variance, axis=1))
    #plt.pcolormesh(variance)
    plt.show()