import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

def implot(array, title=None, colorbar=None):
    if np.iscomplexobj(array):
        array = np.log(np.abs(array))
    plt.figure()
    plt.imshow(array)
    if array.ndim == 3:
        for i in range(array.shape[0]):
            implot(array[i], title[i])

    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.show()


def data2pd(data):
    metric_data = collections.defaultdict(dict)
    for alg in data.keys():
        for prox in data[alg].keys():
            for m in data[alg][prox][1].keys():
                metric_data[m][alg + prox] = data[alg][prox][1][m]

    for m in metric_data.keys():
        metric_data[m] = pd.DataFrame.from_dict(metric_data[m])

    return metric_data


def ssos(I, axis=0):
    """
    Return the square root of the sum of square along axis
    Parameters
    ----------
    I: ndarray
    axis: int
    """
    return np.sqrt(np.sum(np.abs(I) ** 2, axis=axis))


if __name__ == "__main__":
    data = np.load("data/results-online3.npy", allow_pickle=True).tolist()
    data_offline = np.load("data/results-offline-mask3.npy", allow_pickle=True).tolist()
    
    implot(ssos(data["condatvu"]["OWL"][0]))
    
    online = data2pd(data)
    offline = data2pd(data_offline)
    for reg in ["OWL", "GroupLASSO"]:
        for val in ["psnr", "ssim", "offline_cost"]:
            plt.figure()
            plt.xlabel('iterations')
            plt.ylabel(val)
            plt.title(f"{reg}: online/offline")
            plt.plot(online[val]["condatvu" + reg], '-r', label="condatvu" + reg + '-on')
            plt.plot(online[val]["pogm" + reg], '-b', label="pogm" + reg + '-on')
            plt.plot(offline[val]["condatvu" + reg], '--r', label="condatvu" + reg + '-off')
            plt.plot(offline[val]["pogm" + reg], '--b', label="pogm" + reg + '-off')
            plt.legend()
            plt.savefig(f"plot/{reg}_{val}.png")
            plt.show()