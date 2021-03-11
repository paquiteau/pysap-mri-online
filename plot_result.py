import numpy as np

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

def iter_plot(measures, x=None, ylabel="" ):
    if x is None:
        xlabel = "iteration"
    else:
        xlabel = "time"
    fig = plt.figure()
    for y_name, y in measures:
        x = np.arange(len(y))
        plt.plot(x,y,label=y_name)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return fig


if __name__ == "__main__":
    data = np.load("data/results-offline-mask.npy",allow_pickle=True).tolist()
    metric_name = ["time", "cost", "cost_rel", "ssim", "psnr"]
    metric_data = {m:[] for m in metric_name}
    for alg in data.keys():
        for prox in data[alg].keys():
            for m in metric_name:
                metric_data[m].append((alg+prox, data[alg][prox][1].get(m)))
    plots = ["time", "ssim", "psnr"]
    for m in plots:
        print(m)
        if metric_data[m][0][1] is None:
            continue
        print(metric_data[m])
        iter_plot(metric_data[m],ylabel=m)

    fig = plt.figure()
    for y_name, y in metric_data["cost_rel"]:
        x = np.arange(len(y))
        plt.semilogy(x, y, label=y_name)
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("log(cost_resl)")
    plt.show()