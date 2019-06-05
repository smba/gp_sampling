from collections import Counter
from typing import Sequence

import analysis.change_points as cps
import io_handling
import pandas as pd
import matplotlib.pyplot as plt
import metrics
import numpy as np


plt.style.use("ggplot")

class ChangePointEstimation:
    """
    ...
    """
    
    kernels = ["Brownian", "Matern32", "Matern52", "RatQuad", "RBF"]
    
    def __init__(self, subject_system, ground_truth_path):
        self.system_name = subject_system
        self.ground_truth = io_handling.FileLoader.load_ground_truth(ground_truth_path)    
        
        
    def cp_analysis(self, col: int, npz_path: str, kernel: str, training_level: float = 0.01, estimator="binary"):
        """
        @param col: Column index of the loaded ground truth data set
        @param kernel: kernel--- 
        @param npz_path: Absolute path to .npz file 
        """
        col = self.ground_truth.columns[col]
        means, stds = io_handling.FileLoader.load_model_series(npz_path)
        
        # Obtain ground truth for the column
        ground_truth_mean = self.ground_truth[col]
        ground_truth_mean.dropna(inplace=True)  
        ground_truth_mean = ground_truth_mean.values
        
        # Obtain (sort of) ground truth change points data by applying binary segmentation to the 
        # ground truth signal observation. 
        a = cps.BinaryChangePointAnalyzer()
        change_points = a.detect_change_points(ground_truth_mean)
        
        iteration = max(min(int(training_level * ground_truth_mean.shape[0]), means.shape[0] - 1), 1)
        mean = means[iteration]

        if estimator == "binary":
            a = cps.BinaryChangePointAnalyzer()
        elif estimator == "bottomup":
            a = cps.BottomUpChangePointAnalyzer()
            
        cps_from_estimate = a.detect_change_points(mean)
            
        precision, recall = metrics.fuzzy_precall(change_points, cps_from_estimate, fuzzy=5)
        
        return (col, kernel, training_level, precision, recall, estimator)

    def analyze(self, path_template: str): 
        results = []
        for c, col in enumerate(self.ground_truth.columns):
            for kernel in ChangePointEstimation.kernels:
                for training_level in [0.01, 0.03, 0.05, 0.1]:
                    try:
                        npz_path = path_template.format(self.system_name, self.system_name, col, kernel)
                        for estimator in ["binary", "bottomup"]:
                            results.append( self.cp_analysis(c, npz_path, kernel, training_level, estimator) )
                    except FileNotFoundError:
                        print("File not found? ¯\_(ツ)_/¯: " + npz_path)
                
        return results
                    
if __name__ == "__main__":
    cpe = ChangePointEstimation("lrzip", "../../resources/ground_truth/lrzip.csv")
    results = cpe.analyze(path_template="/media/stefan/053F591A314BD654/kernel/{}/{}_{}_{}_uncertainty.npz")
    results = pd.DataFrame(results)
    results.columns = ["variant", "kernel", "training", "precision", "recall", "estimator"]
    print(results.groupby(by = ["kernel", "training", "estimator"]).mean())
    
    pass