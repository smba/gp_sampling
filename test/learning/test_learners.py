'''
Created on Jun 4, 2019

@author: stefan
'''
import unittest
import learning.learners
import numpy as np

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

class Test(unittest.TestCase):


    def testActiveLearner(self):
        n_steps = 10
        s = self.signal
        
        a = learning.learners.ActiveLearner(np.arange(len(s)), s)
        a.iterative_train(max_iter = n_steps)

        self.assertTrue(
            len(a.training_set) == n_steps + 5 + 1, 
            "Training set size: {}, should be: {}".format(len(a.training_set), n_steps + 5 + 1)
        )
        
        self.assertTrue(len(
            a.training_set) == len(set(a.training_set)), 
            "Training set size: {}, should be: {}".format(len(a.training_set), n_steps + 5)
        )
        
    def testBalancedActiveLearner(self):
        n_steps = 15
        s = self.signal
        
        a = learning.learners.BalancedActiveLearner(np.arange(len(s)), s, balance_limit=10)
        a.iterative_train(max_iter = n_steps)

        self.assertTrue(
            len(a.training_set) == 10, 
            "Training set size: {}, should be: {}".format(len(a.training_set), 10)
        )
        
        self.assertTrue(len(
            a.training_set) == len(set(a.training_set)), 
            "Training set size: {}, should be: {}".format(len(a.training_set), 10)
        )
        
    def testIterativeRandomLearner(self):
        n_steps = 10
        s = self.signal
        
        a = learning.learners.ActiveLearner(np.arange(len(s)), s)
        a.iterative_train(max_iter = n_steps)

        self.assertTrue(
            len(a.training_set) == n_steps + 5 + 1, 
            "Training set size: {}, should be: {}".format(len(a.training_set), n_steps + 5 + 1)
        )
        
        self.assertTrue(len(
            a.training_set) == len(set(a.training_set)), 
            "Training set size: {}, should be: {}".format(len(a.training_set), n_steps + 5)
        )


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()