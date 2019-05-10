import jpype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import os

def loadData(filename, get_params = False):
    '''
    Output is a dataframe of 333 regions vs 405 time points
    '''
    if type(filename) is int:
        filename = str(filename)
    if not filename.endswith('.tsv'):
        filename = filename + '.tsv'
    if not filename.startswith('Data'):
        filename = 'Data/' + filename
    df = pd.read_csv(filename, sep = '\t', index_col = 0)
    if get_params:
        param_path = 'Results/AIS Local - Individual Parameters/idx/{}.csv'.format(basename(filename))
        param_df = pd.read_csv(param_path, header = None)
        return df, param_df
    return df

def getAllFiles(path = 'Data', extension = '.tsv'):
    files = []
    for f in os.listdir(path):
        if f.endswith(extension):
            files.append(os.path.join(path, f))
    return sorted(files)

def plotHeatmap(df, divergent = False):
    if divergent:
        vmax = np.abs(df).max()
        im = plt.imshow(df, cmap = 'RdBu', vmax = vmax, vmin = -vmax)
    else:
        im = plt.imshow(df)
    plt.xlabel('Time')
    plt.ylabel('Region')
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.show()

def plotTimeseries(df, idx = 0):
    if isinstance(df, np.ndarray):
        y = pd.DataFrame(df)
    else:
        y = df.iloc[idx].T
    x = range(0, len(y), 50)
    ax = y.plot()
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    plt.xlabel('Time')
    plt.show()

def preprocess(array):
    x = detrend(array)
    return x / np.std(x)

def getMaxIdx2D(array_2d, print_ = True):
    a_len, b_len = array_2d.shape
    max_val = array_2d.max()
    max_idx = array_2d.argmax()
    max_idx = np.unravel_index(max_idx, (a_len, b_len))
    if print_:
        print("Max: {:.4f} at {}".format(max_val, max_idx))
    return max_idx

def startCalc(measure = 'ais', estimator = 'ksg'):
    if not jpype.isJVMStarted():
        if os.path.exists("../noradrenaline/bin/infodynamics.jar"):
            jarLocation = "../noradrenaline/bin/infodynamics.jar"
        else:
            jarLocation = "../../Other Units/1. CSYS5030 - Self Organisation and Criticality/JIDT Toolkit/infodynamics.jar"
        # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
        jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx2048M", "-ea", "-Djava.class.path=" + jarLocation)

    if measure.lower() == 'ais':
        calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
    elif measure.lower() == 'te':
        calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    return calc

def loadArrays(path = 'Results/AIS Local - Individual Parameters', extension = '.csv'):
    files = getAllFiles(path, extension)
    for i, file in enumerate(files):
        df = pd.read_csv(file, header = None, squeeze = True)
        if i == 0:
            array_size = list(df.shape)
            array = np.zeros([len(files)] + array_size)  # for AIS array, size = 333 regions, 405 time-points
        array[i] = df.values
    return array

def getTopIdx1D(array_1d, top_no = 10):
    idx_dict = dict(zip(range(len(array_1d)), array_1d))
    sorted_dict = sorted(idx_dict, reverse = True, key = idx_dict.get)
    return sorted_dict[:top_no]

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

if __name__ == '__main__':
    df = loadData(100307)
