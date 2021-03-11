import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

# plt.close()
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


def iter_plot(measures, x=None, ylabel=""):
    if x is None:
        xlabel = "iteration"
    else:
        xlabel = "time"
    fig = plt.figure()
    for y_name, y in measures:
        x = np.arange(len(y))
        plt.plot(x, y, label=y_name)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return fig


def data2pd(data):
    metric_data = collections.defaultdict(dict)
    for alg in data.keys():
        for prox in data[alg].keys():
            for m in data[alg][prox][1].keys():
                metric_data[m][alg + prox] = data[alg][prox][1][m]

    for m in metric_data.keys():
        metric_data[m] = pd.DataFrame.from_dict(metric_data[m])

    return metric_data


if __name__ == "__main__":
    data = np.load("data/results-online2.npy", allow_pickle=True).tolist()
    data_offline = np.load("data/results-offline-mask2.npy", allow_pickle=True).tolist()

    online = data2pd(data)
    offline = data2pd(data_offline)
    COLORS = ["b","r","g","k"]
    cols = [["condatvuOWL","pogmOWL"],["condatvuGroupLASSO","pogmGroupLASSO"]]
    metrics = list(online.keys())
plt.close()
for col in cols:
    for m in ["psnr","ssim","cost"]:
            print(col)
            plt.figure()
            ax = online[m][col].plot(color= COLORS, xlabel='iteration',ylabel=m)
            offline[m][col].plot(ax=ax,color=COLORS, linestyle='dashed', xlabel='iteration',ylabel=m,title="offline vs online")
            plt.show()