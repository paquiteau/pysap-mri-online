import pandas as pd
class Results:
    def __init__(self, **kwargs):
        for key, v in kwargs.items():
            sr = pd.Series(v['values'], index=v['index'])
            self.__setattr__(key, sr)
    def cost_decrease_percent(self):
        return -100*(self.cost_off.iloc[-1] - self.cost_off.iloc[0])/self.cost_off.iloc[-1]
