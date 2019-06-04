from collections import Counter

import analysis.change_points as cps
import io_handling
import matplotlib.pyplot as plt
import metrics
import numpy as np


plt.style.use("ggplot")

class ChangePointEstimation:
    """
    ...
    """
    
    #ernels = ["Brownian", "Matern32", "Matern52", "RatQuad", "RBF"]
    
    def __init__(self, subject_system, ground_truth_path):
        self.system_name = subject_system
        self.ground_truth = io_handling.FileLoader.load_ground_truth(ground_truth_path)    
        
        
    def cps_analysis(self):
        col = self.ground_truth.columns[13]
        path = "/media/stefan/053F591A314BD654/kernel/{}/{}".format(self.system_name, self.system_name)
        means, stds = io_handling.FileLoader.load_model_series(path + "_" + col + "_Brownian_uncertainty.npz")
        gmean = self.ground_truth[col]
        gmean.dropna(inplace=True)  
        gmean = gmean.values
        
        # extract ground truth change points
        change_points = []
        
        a = cps.BottomUpChangePointAnalyzer()
        change_points += a.detect_change_points(gmean)
        #a = cps.BinaryChangePointAnalyzer()
        #change_points +=a.detect_change_points(gmean)
        #a = cps.WindowChangePointAnalyzer()
        #change_points += a.detect_change_points(gmean)
        #a = cps.SignificanceAnalyzer()
        #change_points += a.detect_change_points(gmean)
        ##a = cps.ConfidenceIntervalAnalyzer()
        #change_points += a.detect_change_points(gmean)
        #a = cps.ThresholdAnalyzer()
        #change_points += a.detect_change_points(gmean)
        
        commons = Counter(change_points)
        commonz = list(filter(lambda k: commons[k] >= 1, commons.keys()))
        if len(gmean) in commonz:
            commonz = commonz[:commonz.index(len(gmean))] + commonz[commonz.index(len(gmean))+1:]
        print(commonz)
        
        precision, recall, f1 = [], [], []
        for i in range(min(means.shape[0], 100)):
            print(i)
            mean = means[i]
            change_points = []
            a = cps.BottomUpChangePointAnalyzer()
            change_points +=a.detect_change_points(mean)
            #a = cps.BinaryChangePointAnalyzer()
            #change_points +=a.detect_change_points(mean)
            #a = cps.WindowChangePointAnalyzer()
            #change_points += a.detect_change_points(mean)
            #a = cps.SignificanceAnalyzer()
            #change_points += a.detect_change_points(mean)
            #a = cps.ConfidenceIntervalAnalyzer()
            #change_points += a.detect_change_points(mean)
            #a = cps.ThresholdAnalyzer()
            #change_points += a.detect_change_points(mean)
            commons = Counter(change_points)
            commons = list(filter(lambda k: commons[k] >= 1, commons.keys()))
            if len(gmean) in commons:
                commons = commons[:commons.index(len(gmean))] + commons[commons.index(len(gmean))+1:]
            
            p, r = metrics.fuzzy_precall(commonz, commons, fuzzy=5)
            try:
                f = 2 * (p*r) / (p + r)
            except ZeroDivisionError:
                f = 0.0
            precision.append(p)
            recall.append(r)
            f1.append(f)
        
        plt.plot(precision, label="precision", linewidth=0.75)
        plt.plot(recall, label="recall", linewidth=0.75)
        plt.plot(f1, label="F1")
        plt.title("{}".format(commonz))
        plt.xlabel("iterations")
        plt.ylabel("%")
        plt.legend()
        plt.show()

        
        
        
if __name__ == "__main__":
    cpe = ChangePointEstimation("xz", "../../resources/ground_truth/xz.csv")
    cpe.cps_analysis()