#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import abstractmethod, Tuple, Sequence

import ruptures

import numpy as np
import pandas as pd
import scipy.stats as stats
 

class ChangePointAnalyzer:
    def __init__(self):
        pass
    
    @abstractmethod
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        ...
     
class CUSUMChangePointAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for the CUSUM algorithm. 
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        Performans change ppoint estimation using CUSUM. If nothing else is specified, auto-tuning 
        the the hyperparameters is used.
        
        @param drift: drift parameter of the CUSUM algorithm
        @param threshold: threshold parameter of the CUSUM algorithm
        :return: list of estimated change points
        '''
        
        if "drift" not in kwargs and "threshold" not in kwargs:
            drift, threshold = self.__tune_parameters(ys)
        elif "drift" in kwargs and "threshold" in kwargs:
            drift = kwargs["drift"]
            threshold = kwargs["threshold"]
        else:
            raise ValueError("Both parameters must either be provided or derived from auto-tuning.")
            
        return list(self.__cusum(ys, drift=drift, threshold=threshold)[0])
    
    def __tune_parameters(self, ys: np.ndarray) -> Tuple[float, float]:
        '''
        Tunes hyperparameters drift and threshold for the CUSUM algorithm.
        
        @param ys:  time-series
        @return: tuple of drift and threshold
        '''
        drift = 0.05 * np.abs(np.nanmax(ys) - np.nanmin(ys)) # 1% change
        threshold = 0.001 * np.nanmean(pd.DataFrame(ys).rolling(window=int(0.05 * ys.shape[0]), center=True).std().values)
        return drift, threshold
    
    def __cusum(self, x, threshold=1, drift=0):
        """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.
        Parameters
        ----------
        x : 1D array_like
            data.
        threshold : positive number, optional (default = 1)
            amplitude threshold for the change in the data.
        drift : positive number, optional (default = 0)
            drift term that prevents any change in the absence of change.
        ending : bool, optional (default = False)
            True (1) to estimate when the change ends; False (0) otherwise.
        show : bool, optional (default = True)
            True (1) plots data in matplotlib figure, False (0) don't plot.
        ax : a matplotlib.axes.Axes instance, optional (default = None).
        Returns
        -------
        ta : 1D array_like [indi, indf], int
            alarm time (index of when the change was detected).
        tai : 1D array_like, int
            index of when the change started.
        taf : 1D array_like, int
            index of when the change ended (if `ending` is True).
        amp : 1D array_like, float
            amplitude of changes (if `ending` is True).
        """
    
        x = np.atleast_1d(x).astype('float64')
        gp, gn = np.zeros(x.size), np.zeros(x.size)
        ta, tai, taf = np.array([[], [], []], dtype=int)
        tap, tan = 0, 0
        amp = np.array([])
        # Find changes (online form)
        for i in range(1, x.size):
            s = x[i] - x[i-1]
            gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
            gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
            if gp[i] < 0:
                gp[i], tap = 0, i
            if gn[i] < 0:
                gn[i], tan = 0, i
            if gp[i] > threshold or gn[i] > threshold:  # change detected!
                ta = np.append(ta, i)    # alarm index
                tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
                gp[i], gn[i] = 0, 0      # reset alarm
                
        return ta, tai, taf, amp
    
class WindowChangePointAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for non-exact, window-based change point estimation using ruptures. 
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        @param width: window size (default is 10)
        @param model: "l1", "rbf", "linear", "normal", "ar" (default is "l2")
        :return: list of estimated change points
        '''
        model = kwargs["model"] if "model" in kwargs else "l2"
        width = kwargs["width"] if "width" in kwargs else 10
        
        estimator = ruptures.Window(width=width, model=model).fit(ys)
        return estimator.predict(pen=1)

class BinaryChangePointAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for non-exact, binary segmentation change point estimation using ruptures. 
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        @param model: "l1", "rbf", "linear", "normal", "ar" (default is "l2")
        :return: list of estimated change points
        '''
        model = kwargs["model"] if "model" in kwargs else "l2"
        estimator = ruptures.Binseg(model=model).fit(ys)
        
        return estimator.predict(pen=3)
    
class BottomUpChangePointAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for non-exact, bottom-up change point estimation using ruptures. 
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        @param model: "l1", "rbf", "linear", "normal", "ar" (default is "l2")
        :return: list of estimated change points
        '''
        model = kwargs["model"] if "model" in kwargs else "l2"
        estimator = ruptures.BottomUp(model=model).fit(ys)
        
        return estimator.predict(pen=3)

class ConfidenceIntervalAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for change point estimation using confidence interval overlap (significance)
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        Identifies performance anomalies as such revisions, for which performance is not contained in
        a confidence interval (CI) specified by preceding commits.
        The default z-score is 5, the window size is 10.
        
        @param window: sliding window size
        @param z: default is 5 sigma
        :return: list of estimated change points
        '''
        
        z = kwargs["z"] if "z" in kwargs else 5
        window = kwargs["window"] if "window" in kwargs else 10
        ys = pd.DataFrame(ys)
        
        rolling = ys.rolling(window=window)
        ci = lambda s: s[-1] <= np.mean(s[:-1]) + z * np.std(s[:-1]) and s[-1] >= np.mean(s[:-1]) - z * np.std(s[:-1])
        sigs = rolling.apply(ci, raw=True)

        return list(sigs != 1.0)

class SignificanceAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for change point estimation using the Mann-Whitney-U significance test
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        Identifies performance anomalies as such revisions, for which performance is [statistically] significantly
        different compared to preceeding revisions using the Mann-Whitney-U test (Wilcoxon rank sum test).
        
        @param window: number of preceeding commits to compare current performance against
        @param p: significance level, default is 0.05
        :return: list of estimated change points
        '''
        p = kwargs["p"] if "p" in kwargs else 0.05
        window = kwargs["window"] if "window" in kwargs else 10
        ys = pd.DataFrame(ys)
        
        rolling = ys.rolling(window=window, center=True)
        sig = lambda s: stats.mannwhitneyu(s[:int(window/2)], s[int(window/2):]).pvalue
        sigs = rolling.apply(sig)
        return list(sigs[sigs < p].dropna())
    
class ThresholdAnalyzer(ChangePointAnalyzer):
    '''
    Wrapper class for change point estimation using threshold deviation.
    '''
    def __init__(self):
        ChangePointAnalyzer.__init__(self)
        
    def detect_change_points(self, ys: np.ndarray, **kwargs) -> Sequence[int]:
        '''
        Identifies performance anomalies as such revisions, for which performance increases/decreases
        by more than a relative threshold (default is 1%) from the preceeding revisions (default is 10).

        :param window: number of preceding commits to compare current performance against
        :param threshold: critical performance change threshold
        :return: list of estimated change points
        '''
        threshold = kwargs["threshold"] if "threshold" in kwargs else 0.01
        window = kwargs["window"] if "window" in kwargs else 10
        ys = pd.DataFrame(ys)
        
        means = ys.shift(1).rolling(window=window).mean()
        change = (ys - means) / ys
        change = change.abs()
        return list(change > threshold)

def segment(signal):
    if signal.shape[0] < 2:
        return 0
    found = 0
    change_point = pettitt_test(signal)
    if change_point[1] <= 0.05:
        found += 1
        found += segment(signal[:change_point[0]])
        found += segment(signal[change_point[0]:])
        return found
    else:
        return 0

def pettitt_test(signal):
    signal = signal.values
    T = signal.shape[0]
    U = []
    for t in range(T):
        res = 0
        for i in range(t):
            for j in range(t+1, T):
                res += np.sign(signal[i] - signal[j])
        U.append(res)
    loc = np.argmax(np.abs(U))
    K = np.max(np.abs(U))
    p = 2 * np.exp(-6*K**2/(T**3+T**2))

    #direction
    cp_range = np.abs(signal[loc - 1] - signal[loc + 1])
    if signal[loc - 1] > signal[loc + 1]:
        direction = "decreasing"
    elif signal[loc - 1] < signal[loc+ 1]:
        direction = "increasing"
    else:
        direction = "no trend"
    return {
        "change_point": p < 0.05, 
        "change_point_location": loc, 
        "change_point_direction": direction, 
        "change_point_range": cp_range * 1000, 
        "change_point_p": p
    }

