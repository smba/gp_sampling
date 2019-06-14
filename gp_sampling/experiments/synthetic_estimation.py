import random 
import numpy as np

def create_signal():
    cps = np.random.choice(list(range(5000)), size=random.randint(1,50))
    sig = [0]
    for i in range(4999):
        s = sig[-1]
        if i in cps:
            s += np.random.normal(0, 3)
        s += 0.3 * np.random.normal(0, 1)
        sig.append(s)
    return np.array(sig), cps

if __name__ == "__main__":
    
    signal, cps = create_signal()
    