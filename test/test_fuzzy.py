'''
Created on Jun 5, 2019

@author: stefan
'''
import unittest
import analysis.change_points as cps
import metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from collections import Counter
import seaborn as sns
#plt.style.use("ggplot")
sns.set_style("whitegrid")

def signal(seed=123):
    np.random.seed(seed)
    cps = np.random.choice(list(range(1000)), size=random.randint(3,30))
    sig = [0]
    for i in range(999):
        s = sig[-1]
        if i in cps:
            s += np.random.normal(0, 3)
        s += 0.3 * np.random.normal(0, 1)
        sig.append(s)
    return np.array(sig), cps

class Test(unittest.TestCase):

    def setUp(self):
        pass
        


    def tearDown(self):
        pass


    def testName(self):
        p, r = [], []
        pa, pb, pc, ra, rb, rc = [],[],[],[],[],[]
        for s in range(100):
            print(s)
            self.signal, self.cp = signal(s)
            change_points = []
            a = cps.CUSUMChangePointAnalyzer()
            b = cps.BinaryChangePointAnalyzer()
            c = cps.WindowChangePointAnalyzer()
            ca = a.detect_change_points(self.signal)
            cb = b.detect_change_points(self.signal)
            cc = c.detect_change_points(self.signal)
            change_points += ca
            change_points += cb
            change_points += cc
            
            change_points.sort()
            
            cpf = Counter(change_points)
            N = 5
            keks = []
            for key in cpf:
                # get keys that are +-N of key
                keys = list(filter(lambda k: k < key + N and k > key - N, cpf))
                
                kek = []
                for keyx in keys:
                    kek += cpf[keyx] * [keyx]
                if len(kek) > 1:
                    keks.append(int(np.median(kek)))
                    
            keks = list(dict.fromkeys(keks))
    
            change_points = list(set(change_points[:-3]) - set(keks))
            
            x, y = metrics.fuzzy_precall(self.cp, keks, fuzzy=5)
            p.append(x)
            r.append(y)
            x, y = metrics.fuzzy_precall(self.cp, ca, fuzzy=5)
            pa.append(x)
            ra.append(y)
            
            x, y = metrics.fuzzy_precall(self.cp, cb, fuzzy=5)
            pb.append(x)
            rb.append(y)
            
            x, y = metrics.fuzzy_precall(self.cp, cc, fuzzy=5)
            pc.append(x)
            rc.append(y)
            
        #plt.hist(p, bins=35, alpha=0.5, label='precision', edgecolor='black', linewidth=0.5)
        #plt.hist(r, bins=35, alpha=0.5, label='recall', edgecolor='black', linewidth=0.5)
        print(np.mean(p), np.median(p), np.std(p))
        print(np.mean(r), np.median(r), np.std(r))
        
        x = range(10)
        y = range(10)
        
        fig = plt.figure()

        plt.subplot(2, 2, 1)
        plt.title("cumulative sum")
        sns.distplot(pa, hist=False, rug=True, label="precision (median: {})".format(round(np.mean(pa), 2)))
        sns.distplot(ra, hist=False, rug=True, label="recall (median: {})".format(round(np.mean(ra), 2)))
        plt.xlim((0,1))
        plt.xlabel("precision or recall")
        plt.ylabel("frequency")

        
        plt.subplot(2, 2, 2)
        plt.title("Binary")
        sns.distplot(pb, hist=False, rug=True, label="precision (median: {})".format(round(np.mean(pb), 2)))
        sns.distplot(rb, hist=False, rug=True, label="recall (median: {})".format(round(np.mean(rb), 2)))
        plt.xlim((0,1))
        plt.xlabel("precision or recall")
        plt.ylabel("frequency")

        
        plt.subplot(2, 2, 3)
        plt.title("Window")
        sns.distplot(pc, hist=False, rug=True, label="precision (median: {})".format(round(np.mean(pc), 2)))
        sns.distplot(rc, hist=False, rug=True, label="recall (median: {})".format(round(np.mean(rc), 2)))
        plt.xlabel("precision or recall")
        plt.ylabel("frequency")
        plt.xlim((0,1))

        
        plt.subplot(2, 2, 4)
        plt.title("Merged")
        sns.distplot(p, hist=False, rug=True, label="precision (median: {})".format(round(np.mean(p), 2)))
        sns.distplot(r, hist=False, rug=True, label="recall (median: {})".format(round(np.mean(r), 2)))
        plt.xlim((0,1))
        plt.xlabel("precision or recall")
        plt.ylabel("frequency")

        plt.show()
        
        #plt.plot(self.signal, linewidth=0.75, color="black")
        #plt.scatter(keks[:-1], self.signal[keks[:-1]], color="crimson", marker="s")
        #p##lt.scatter(change_points, self.signal[change_points], color="green", marker="s", alpha=0.4)
        #plt.show()
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()