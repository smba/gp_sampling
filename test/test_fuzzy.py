'''
Created on Jun 5, 2019

@author: stefan
'''
import unittest
import analysis.change_points as cps
import metrics
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#plt.style.use("ggplot")

def signal(seed=123):
    np.random.seed(seed)
    minor = np.random.choice(list(range(1000)), size=15)
    major = np.random.choice(list(range(1000)), size=5)
    sig = [0]
    for i in range(999):
        s = sig[-1]
        if i in major:
            s += 1.5 *np.random.normal(0, 3)
        if i in minor:
            s += 1.5 * np.random.normal(0,1)
        s += 0.15 * np.random.normal(0, 1)
        sig.append(s)
    return np.array(sig), list(np.append(minor, major))

class Test(unittest.TestCase):

    def setUp(self):
        pass
        


    def tearDown(self):
        pass


    def testName(self):
        p, r = [], []
        pa, pb, pc, ra, rb, rc = [],[],[],[],[],[]
        for s in range(0, 2000):
            print(s)
            self.signal, self.cp = signal(s)
            change_points = []
            a = cps.BottomUpChangePointAnalyzer()
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
                if len(kek) > 2:
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
        plt.title("Bottom-Up")
        plt.hist(pa, alpha=0.5, edgecolor="black", linewidth=0.5, label="precision", bins=50)
        plt.hist(ra, edgecolor="black", linewidth=0.5, label="recall", alpha=0.5, bins=50)
        plt.xlim((0,1))
        
        plt.subplot(2, 2, 2)
        plt.title("Binary")
        plt.hist(pc, edgecolor="black", linewidth=0.5, label="precision", alpha=0.5, bins=50)
        plt.hist(rb, edgecolor="black", linewidth=0.5, label="recall", alpha=0.5, bins=50)
        plt.xlim((0,1))
        
        plt.subplot(2, 2, 3)
        plt.title("Window")
        plt.hist(pc, edgecolor="black", linewidth=0.5, label="precision", alpha=0.5, bins=50)
        plt.hist(rc, edgecolor="black", linewidth=0.5, label="recall", alpha=0.5, bins=50)
        plt.xlim((0,1))
        
        plt.subplot(2, 2, 4)
        plt.title("Merged")
        plt.hist(p, edgecolor="black", linewidth=0.5, label="precision", alpha=0.5, bins=50)
        plt.hist(r, edgecolor="black", linewidth=0.5, label="recall", alpha=0.5, bins=50)
        plt.xlim((0,1))
        plt.legend()
        plt.show()
        
        #plt.plot(self.signal, linewidth=0.75, color="black")
        #plt.scatter(keks[:-1], self.signal[keks[:-1]], color="crimson", marker="s")
        #p##lt.scatter(change_points, self.signal[change_points], color="green", marker="s", alpha=0.4)
        #plt.show()
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()