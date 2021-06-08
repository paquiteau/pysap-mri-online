import pandas as pd
import matplotlib as plt

class Results:
    def __init__(self, **kwargs):
        for key, v in kwargs.items():
            sr = pd.Series(v['values'], index=v['index'])
            self.__setattr__(key, sr)

    def plot(self, attr='psnr', ax=None, ):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.__getattribute__(attr))
