#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import random
import matplotlib.pyplot as plt
import numpy as np
import analysis.change_points as cps
import learning.learners
import metrics
import scipy.stats as stats
import ruptures
import learning

def signal():
    np.random.seed(123)
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
    
    
    def testChangePointEstimation(self):
        s, cps = signal()
        a = learning.learners.IterativeRandomLearner(np.arange(len(s)), s)
        a.iterative_train(max_iter=20)
        mean, std = a.predict()

        plt.plot(np.arange(len(s)), s)
        plt.plot(np.arange(len(s)), mean)
        plt.show()
         
    def testName(self):
        algo1 = cps.BinaryChangePointAnalyzer(self.signal)
        result = algo1.detect_change_points(self.signal)[:-1]
        plt.scatter(result, self.signal[result], color="dodgerblue", marker="X", s=120)
        print(metrics.fuzzy_precall(self.cps, result, fuzzy=5))
        
        algo1 = cps.WindowChangePointAnalyzer(self.signal)
        result = algo1.detect_change_points(self.signal)[:-1]
        plt.scatter(result, self.signal[result]+0.2, color="blue", marker="X", s=120)
        print(metrics.fuzzy_precall(self.cps, result, fuzzy=5))

    def test_BinaryChangePointAnalyzer(self):
        algo1 = cps.BinaryChangePointAnalyzer()

        sig, cpss = signal()
        change_points = algo1.detect_change_points(sig)[:-1]
        pre, rec = metrics.fuzzy_precall(cpss, change_points, fuzzy=5)

        self.assertTrue(np.median(pre) > 0.6, "precision {} is smaller than 0.6".format(np.median(pre)))
        self.assertTrue(np.median(rec) > 0.8, "recall {} is smaller than 0.8".format(np.median(rec)))

    def test_WindowChangePointAnalyzer(self):
        algo1 = cps.WindowChangePointAnalyzer()
        
        sig, cpss = signal()
        change_points = algo1.detect_change_points(sig)[:-1]
        pre, rec = metrics.fuzzy_precall(cpss, change_points, fuzzy=5)

        self.assertTrue(np.median(pre) > 0.6, "precision {} is smaller than 0.6".format(np.median(pre)))
        self.assertTrue(np.median(rec) > 0.8, "recall {} is smaller than 0.8".format(np.median(rec)))
        
    def test_BottomUpChangePointAnalyzer(self):
        algo1 = cps.BottomUpChangePointAnalyzer()
        
        sig, cpss = signal()
        change_points = algo1.detect_change_points(sig)[:-1]
        pre, rec = metrics.fuzzy_precall(cpss, change_points, fuzzy=5)

        self.assertTrue(np.median(pre) > 0.6, "precision {} is smaller than 0.6".format(np.median(pre)))
        self.assertTrue(np.median(rec) > 0.8, "recall {} is smaller than 0.8".format(np.median(rec)))

    def test_CUSUMChangePointAnalyzer(self):
        algo1 = cps.CUSUMChangePointAnalyzer()
        
        sig, cpss = signal()
        change_points = algo1.detect_change_points(sig)[:-1]
        pre, rec = metrics.fuzzy_precall(cpss, change_points, fuzzy=5)

        self.assertTrue(np.median(pre) > 0.6, "precision {} is smaller than 0.6".format(np.median(pre)))
        self.assertTrue(np.median(rec) > 0.8, "recall {} is smaller than 0.8".format(np.median(rec)))

if __name__ == "__main__":
    plt.style.use('ggplot')
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()