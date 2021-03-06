import numpy as np
import pandas as pd

class FileLoader:
    
    def __init__(self):
        pass
    
    @classmethod
    def load_model_series(cls, path) -> np.ndarray:
        unpckld = np.load(path)
        return unpckld['mean'], unpckld['stds']
    
    @classmethod
    def load_ground_truth(cls, path) -> pd.DataFrame:
        gt = pd.DataFrame.from_csv(path)
        return gt