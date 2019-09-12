import unittest
import os
import pandas as pd
import io_handling

class TestA(unittest.TestCase):

    def test_load_ground_truth(self):
        prefix = "../resources/ground_truth/"
        for file in os.listdir(prefix):
            df = io_handling.FileLoader.load_ground_truth(prefix + file)

    def test_load_model_series(self):
        file = "../resources/xz_24_RBF_uncertainty.npz"
        file = io_handling.FileLoader.load_model_series(file)
        print(len(file))
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()