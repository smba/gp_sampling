#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import random
import matplotlib.pyplot as plt
import numpy as np
import analysis.change_points as cps
import metrics
import scipy.stats as stats
import ruptures

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