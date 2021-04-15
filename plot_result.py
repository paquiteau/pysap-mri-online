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

def ssos(I, axis=0):
    """
    Return the square root of the sum of square along axis
    Parameters
    ----------
    I: ndarray
    axis: int
    """
    return np.sqrt(np.sum(np.abs(I) ** 2, axis=axis))

def pandify(data):
    metric_data = collections.defaultdict(dict)
    for alg in data.keys():
        for prox in data[alg].keys():
            for m in data[alg][prox][1].keys():
                metric_data[m][alg + prox] = data[alg][prox][1][m]
    for m in metric_data.keys():
        metric_data[m] = pd.DataFrame(metric_data[m])
    return metric_data

if __name__ == '__main__':
    version = 9
    data = np.load(f"data/results-online{version}.npy", allow_pickle=True).tolist()
    metric_data = pandify(data)
    offline_data = np.load(f"data/results-offline{version}.npy", allow_pickle=True).tolist()
    offline_metrics = pandify(offline_data)

    x_online = data['condatvu']['GroupLASSO'][0]
    x_offline = offline_data['condatvu']['GroupLASSO'][0]

# raw comparison of online et offline results
    plt.figure()
    plt.imshow(np.fft.fftshift(ssos(x_online - x_offline)))
    plt.colorbar()
    plt.show()


# cost plots
    for reg in ['GroupLASSO']:
        for m in ['grad', 'sum']:
            plt.figure()
            plt.xlabel("iteration")
            plt.ylabel(m)
            plt.semilogy(metric_data[m + '_on']['condatvu' + reg], '-r', label="condatvu" + reg + '-on')
            plt.semilogy(offline_metrics[m + '_off']['condatvu' + reg], '-k', label="condatvu" + reg + '-off')
            #plt.semilogy(metric_data[m + '_on']['pogm' + reg], '-b', label="pogm" + reg + '-on')
            plt.semilogy(metric_data[m + '_off']['condatvu' + reg], '-b', label="condatvu" + reg + '-onoff')
            #plt.semilogy(metric_data[m + '_off']['pogm' + reg], '--y', label="pogm" + reg + '-onoff')
            #plt.semilogy(offline_metrics[m + '_off']['pogm' + reg], ':b', label="pogm" + reg + '-off')
            plt.legend()
            plt.savefig(f'plot/{reg}_{m}')
            plt.show()
# metrics
    for reg in ['GroupLASSO']:
        for m in ['psnr']:
            plt.figure()
            plt.xlabel("iteration")
            plt.ylabel(m)
            plt.semilogy(metric_data[m]['condatvu' + reg], '-r', label="condatvu" + reg + '-on')
            # plt.semilogy(metric_data[m]['pogm' + reg], '-b', label="pogm" + reg + '-on')
            plt.semilogy(offline_metrics[m]['condatvu' + reg], ':r', label="condatvu" + reg + '-off')
            # plt.semilogy(offline_metrics[m]['pogm' + reg], ':b', label="pogm" + reg + '-off')
            plt.legend()
            plt.savefig(f'plot/{reg}_{m}')
            plt.show()

reg = "GroupLASSO"
m = "grad"
quot = offline_metrics[m + '_off']['condatvu' + reg]/metric_data[m + '_on']['condatvu' + reg]
plt.figure()
plt.semilogy(quot)
plt.title(f"condatvuOWL ratio :{np.mean(quot[-30:])}")
plt.ylabel("grad_on/grad_off")
plt.show()

