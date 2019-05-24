#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ruptures as rpt
from typing import abstractmethod, Tuple, Sequence

class ChangePointAnalyzer:
    def __init__(self, ys: np.ndarray):
        self.ys = ys
    
    @abstractmethod
    def detect_change_points(self) -> Tuple[Sequence[int], Sequence[float]]:
        ...
     
class CUSUMChangePointAnalyzer(ChangePointAnalyzer):
    def __init__(self, ys: np.ndarray):
        super.__init__(self, ys)
        
    def detect_change_points(self) -> Tuple[Sequence[int], Sequence[float]]:
        pass # TODO
    
class WindowChangePointAnalyzer(ChangePointAnalyzer):
    def __init__(self, ys: np.ndarray):
        super.__init__(self, ys)
        
    def detect_change_points(self) -> Tuple[Sequence[int], Sequence[float]]:
        pass # TODO

class BinaryChangePointAnalyzer(ChangePointAnalyzer):
    def __init__(self, ys: np.ndarray):
        super.__init__(self, ys)
        
    def detect_change_points(self) -> Tuple[Sequence[int], Sequence[float]]:
        pass # TODO
    
class BottomUpChangePointAnalyzer(ChangePointAnalyzer):
    def __init__(self, ys: np.ndarray):
        super.__init__(self, ys)
        
    def detect_change_points(self) -> Tuple[Sequence[int], Sequence[float]]:
        pass # TODO

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
    
def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
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
