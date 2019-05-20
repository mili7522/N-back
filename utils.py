import os
import h5py
import jpype
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, detrend, filtfilt
from scipy.stats import zscore


def loadData(filename, path = None, get_params = False, param_path = None, subject_id = None):
    """
    Loads the data (and the corresponding history length and delay if get_params is True)

    Arguments:
        filename -- Name (and path) of the data file to be loaded
        path -- Location of the file, or None (if the filename argument already contains the path)
        get_params -- If True, load the history length and delay corresponding to the file (found by maximising AIS)
        param_path
        subject_id -- Used for the GRP data. Selects which subject's data to load

    Output:
        df -- Dataframe of regions of interest (ROI) vs time points
        params_df
    """
    if path is not None:
        filename = os.path.join(path, filename)
    if filename.endswith('.tsv'):  # The HCP data is made of 493 tsv files of 333 regions (rows) vs 405 time points (columns)
        df = pd.read_csv(filename, sep = '\t', index_col = 0)
    if filename.endswith('.csv'):  # The ATX data is made of 38 csv files (19 subjects with two files each) of 375 regions (rows) vs 286 time points (columns)
        df = pd.read_csv(filename, header = None, index_col = None)
    if filename.endswith('.mat'):  # DONE data or GRP data
        if subject_id is None:
            # The DONE data is made up of 28 mat files of 375 regions (matlab rows) vs 186 time points (matlab columns)
            data = h5py.File(filename, 'r')
            data = data['ts_matrix']  # Gives a HDF5 dataset. Can also get the array using data.value
            df = pd.DataFrame(np.array(data).T)  # Need to transpose it since matlab uses column-major order and numpy uses row-major order
        else:
            # The GRP data is organised as 375 regions x 1940 time points x 100 subjects
            data = loadmat(filename)
            data = data['data_grp_full']
            data = data[:,:,subject_id]
            df = pd.DataFrame(data)
    if get_params:
        assert param_path is not None, "Need to provide a path where parameters are stored"
        # eg param_path = Results/AIS Local - Individual Parameters/idx/
        param_file = os.path.join(param_path, '{}.csv'.format(basename(filename)))
        param_df = pd.read_csv(param_file, header = None)
        return df, param_df
    return df

def getAllFiles(path = 'Data', extension = '.tsv'):
    """
    Returns a list of files in a particular folder that ends with a particular extension
    """
    files = []
    for f in os.listdir(path):
        if f.endswith(extension):
            files.append(os.path.join(path, f))
    return sorted(files)

def plotHeatmap(df, divergent = False, divergent_cmap = 'RdBu', vmax = None, vmin = None):
    if divergent:
        vmax = vmax or np.abs(df).max()
        vmin = vmin or -vmax
        im = plt.imshow(df, aspect = 'auto', cmap = divergent_cmap, vmax = vmax, vmin = vmin)
    else:
        im = plt.imshow(df, aspect = 'auto')
    plt.xlabel('Time')
    plt.ylabel('Region')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

def plotTimeseries(data, region_idx = None):
    """
    Arguments:
        data -- Numpy array of 1D or 2D or pandas DataFrame. If 2D, it is in the shape (regions, time)
        region_idx -- The index of the region to plot, as an Int, or a list of indexs
    """
    assert data.ndim == 1 or region_idx is not None
    if isinstance(data, pd.DataFrame):
        y = data.iloc[region_idx].values
    elif data.ndim == 2:
        y = data[region_idx]
    else:
        y = data
    if y.ndim == 2:  # This is the case if region_idx is a list 
        y = y.T
    plt.plot(y)
    plt.xlim([0, len(y)])
    plt.xlabel('Time')
    if y.ndim == 2:
        plt.legend(labels = region_idx)
    plt.show()

def preprocess(data, sampling_interval = None, sampling_rate = None, apply_global_mean_removal = True, trim_start = 0, trim_end = 0, **filter_params):
    """
    Applies preprocessing to the fMRI data, following a pipeline that will make it suitable for AIS / TE analysis

    Arguments:
        data -- Numpy array or pandas DataFrame of shape (regions, time)
        sampling_interval -- None, or the interval between samples in seconds
        sampling_rate -- None, or the number of samples per second
        apply_global_mean_removal -- If True, global mean removal will be applied to the data
        trim_start -- Number of timesteps to cut off the start of the data to counteract the effect of filtering
        trim_end -- Number of timesteps to cut off at the end of the data to counteract the effect of filtering
        filter_params -- Other filter parameters (fcuthigh and order)

    Output:
        data -- Numpy array after preprocessing: z-scoring, detrending, high pass filtering and applying global mean removal.
                Some starting and ending timepoints may be trimmed
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = zscore(data, axis = 1, ddof = 1)
    data = detrend(data, axis = 1)
    data = high_pass_filter(data, sampling_interval, sampling_rate, **filter_params)
    if apply_global_mean_removal:
        data = data - np.mean(data, axis = 0, keepdims = True)
    # Trim start and end
    data = data[:, trim_start:]
    if trim_end > 0:
        data = data[:, :-trim_end]
    return data

def high_pass_filter(data, sampling_interval = None, sampling_rate = None, fcuthigh = 0.01, order = 3):
    """
    Returns the data after applying a high pass butterworth filter using the filtfilt function

    Arguments:
        data -- Numpy array in the shape (ROI, time)
        sampling_interval -- Seconds per sample
        sampling_rate -- Samples per second, eg. 1.3 samples per second for HCP data
        fcuthigh -- Cutoff frequency (Hz)
        order -- Order of the filter

    Output:
        data -- Numpy array containing the data after filtering is performed
    """
    # Setting up the filter
    assert sampling_interval is not None or sampling_rate is not None
    sampling_rate = sampling_rate or 1 / sampling_interval
    Wn = fcuthigh / ( 1/2 * sampling_rate )
    [b, a] = butter( order, Wn, 'high')

    # Applying the filter
    data = filtfilt( b, a, data, axis = 1, padtype = 'odd', padlen = 3 * ( max(len(a),len(b)) - 1) )
    return data

def acf(data, axis = 1):
    """
    Calculates the autocorrelation coefficients

    Arguments:
        data -- Numpy array of 1d or 2d
        axis -- The number of the axis corresponding to the time dimension
    
    Outputs:
        acf_coeffs -- Numpy array of 1d or 2d. If it's 2d, it is always returned in the shape (lag, ROI)
    """
    data = np.atleast_2d(data)
    n = data.shape[axis]
    mean = np.mean(data, axis = axis, keepdims = True)
    var = np.sum((data - mean) ** 2, axis = axis)  # Unscaled variance
    def r(h):
        if axis == 0:
            acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum(axis = axis) / var
        elif axis == 1:
            acf_lag = ((data[:,:n - h] - mean) * (data[:,h:] - mean)).sum(axis = axis) / var
        return acf_lag
    x = np.arange(n)  # Avoiding lag 0 calculation
    acf_coeffs = np.array(list(map(r, x)))
    return acf_coeffs

def acl(data, axis = 1):
    """
    Calculates the autocorrelation length as expected by the dynamic correlation exclusion parameter in JIDT
    """
    acf_coeffs = acf(data, axis)
    auto_correlation_lengths = np.ceil( 2 * np.sum( acf_coeffs ** 2, axis = 0) ) - 2  # Always sum over the 0th axis
    auto_correlation_lengths = auto_correlation_lengths.astype(int)
    return np.squeeze(auto_correlation_lengths)

def getMaxIdx2D(array_2d, print_ = True):
    """
    Finds the (2D) index of the maximum value in a 2D array

    Arguments:
        array_2d -- 2D Numpy array
        print_ -- If True, the maximum value and index is printed
    
    Returns:
        max_idx -- A tuple of (row, col), that gives the location of the maximum value in the array
    """
    a_len, b_len = array_2d.shape
    max_val = array_2d.max()
    max_idx = array_2d.argmax()
    max_idx = np.unravel_index(max_idx, (a_len, b_len))
    if print_:
        print("Max: {:.4f} at {}".format(max_val, max_idx))
    return max_idx

def startCalc(measure = 'ais', estimator = 'ksg'):
    """
    Start the JIDT calculator
    """
    if not jpype.isJVMStarted():
        if os.path.exists("../noradrenaline/bin/infodynamics.jar"):
            jarLocation = "../noradrenaline/bin/infodynamics.jar"
        else:
            jarLocation = "/home/mike/Downloads/JIDT/infodynamics.jar"
        # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
        jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx2048M", "-ea", "-Djava.class.path=" + jarLocation)

    if measure.lower() == 'ais':
        if estimator.lower() == 'ksg':
            calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
    elif measure.lower() == 'te':
        if estimator.lower() == 'ksg':
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
    df_HCP = loadData('100307.tsv', path = '../Data')
    def checkPreprocessing():
        # Checking comparisons with matlab preprocessing
        from scipy.io import loadmat
        comparison_data = loadmat('../Preprocessing_steps_100307.mat')
        data = preprocess(df_HCP.values, sampling_rate = 1.3)
        print("Close after mean removal?", np.allclose(data.T, comparison_data['data_meanremoval']))
        print("Close acl after filtering?", np.allclose(acl(data, axis = 1), comparison_data['acl_meanremoval'].ravel()))
    checkPreprocessing()
    # Load ATX data
    df_ATX = loadData('STX0001-01_results.csv', path = '/media/mike/Files/Data and Results/N-back/Data/ATX_data')
    # Load DONE data
    df_DONE = loadData('R01A.mat', path = '/media/mike/Files/Data and Results/N-back/Data/DONE_data')
    # Load GRP data
    df_GRP = loadData('data_grp_full.mat', path = '/media/mike/Files/Data and Results/N-back/Data/data_grp_full', subject_id = 0)