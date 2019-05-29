#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import random
import matplotlib.pyplot as plt
import numpy as np
import analysis.change_points as cps
import metrics
import ruptures

def signal():
    minor = [200, 270, 340, 630, 690, 880]
    major = [420, 750, 120]
    sig = [0]
    for i in range(999):
        s = sig[-1]
        if i in major:
            s += 1.5 *np.random.normal(0, 3)
        if i in minor:
            s += 1.5 * np.random.normal(0,1)
        s += 0.05 * np.random.normal(0, 1)
        sig.append(s)
    return np.array(sig), minor + major

class TestCPAnalysis(unittest.TestCase):


    def setUp(self):
        self.signal, self.cps = signal()


    def tearDown(self):
        pass


    def testName(self):
        algo1 = cps.BinaryChangePointAnalyzer(self.signal)
        result = algo1.detect_change_points(self.signal)[:-1]
        plt.scatter(result, self.signal[result], color="dodgerblue", marker="X", s=120)
        print(metrics.fuzzy_precall(self.cps, result, fuzzy=5))
        
        algo1 = cps.WindowChangePointAnalyzer(self.signal)
        result = algo1.detect_change_points(self.signal)[:-1]
        plt.scatter(result, self.signal[result]+0.2, color="blue", marker="X", s=120)
        print(metrics.fuzzy_precall(self.cps, result, fuzzy=5))
        
        algo1 = cps.BottomUpChangePointAnalyzer(self.signal)
        result = algo1.detect_change_points(self.signal)[:-1]
        plt.scatter(result, self.signal[result]+0.4, color="violet", marker="X", s=120)
        print(metrics.fuzzy_precall(self.cps, result, fuzzy=5))
        
        plt.plot(range(len(self.signal)), self.signal)
        
        plt.show()
        

if __name__ == "__main__":
    plt.style.use('ggplot')
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()