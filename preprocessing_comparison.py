from scipy.io import loadmat
from scipy.stats import zscore
from scipy.signal import detrend, butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

transpose_data = False  # Reading in matlab using dlmread transposes the file compared to numpy because matlab uses column-major order while numpy uses row-major

def plot(data, vmin = -1.5, vmax = 1.5):
    plt.imshow(data.T if transpose_data else data, vmin = vmin, vmax = vmax)
    plt.xlabel('Time')
    plt.ylabel('Regions')
    plt.colorbar()
    plt.show()

# Load data
comparison_data = loadmat('../Preprocessing_steps_100307.mat')
# data = comparison_data['data']
data = pd.read_csv('../Data/100307.tsv', sep = '\t', index_col = 0).values
if transpose_data: data = data.T  # Matlab reads it (333,405), while pandas reads it (405,333)
plot(data, vmin = None, vmax = None)

# Z score
comparison_zscore = comparison_data['data_zscore']
data_zscore = zscore(data, axis = 0 if transpose_data else 1, ddof = 1)  # ddof is the degree of freedom correction. Matlab calculates the standard deviation using N-1
print("Close after z-score?", np.allclose(data_zscore if transpose_data else data_zscore.T, comparison_zscore))
plot(data_zscore)

# Detrend
comparison_detrend = comparison_data['data_detrend']
data_detrend = detrend(data_zscore, axis = 0 if transpose_data else 1)  # axis is -1 by default
print("Close after detrending?", np.allclose(data_detrend if transpose_data else data_detrend.T, comparison_detrend))
plot(data_detrend)

# Filtering -- Setting up the filter
comparison_b = comparison_data['b']
comparison_a = comparison_data['a']
fcuthigh = 0.01  # Hz 
sampling_rate = 1.3  # samples per second
Wn = fcuthigh / ( 1/2 * sampling_rate )
order = 3
[b, a] = butter( order, Wn, 'high')
print("Filter value b is close?", np.allclose(b, comparison_b))
print("Filter value a is close?", np.allclose(a, comparison_a))

# Filtering
comparison_filter = comparison_data['data_filter']
data_filter = filtfilt( b, a, data_detrend, axis = 0 if transpose_data else 1, padtype = 'odd', padlen = 3*(max(len(a), len(b)) - 1))  # Scipy's default padlen is 3*max(len(a), len(b)). See https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
print("Close after filtering?", np.allclose(data_filter if transpose_data else data_filter.T, comparison_filter))
plot(data_filter)

# Global mean removal
comparison_meanremoval = comparison_data['data_meanremoval']
data_meanremoval = data_filter - np.mean(data_filter, axis = 1 if transpose_data else 0, keepdims = True)
print("Close after mean removal?", np.allclose(data_meanremoval if transpose_data else data_meanremoval.T, comparison_meanremoval))
plot(data_meanremoval)

### Autocorrelation
def acf(data, axis = 1):
    """
    Adapted from https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python and https://au.mathworks.com/matlabcentral/fileexchange/30540-autocorrelation-function-acf
    See also https://stackoverflow.com/questions/36038927/whats-the-difference-between-pandas-acf-and-statsmodel-acf
    
    Inputs:
        data -- Numpy array of 1d or 2d
        axis -- Time axis
    
    Outputs:
        acf_coeffs -- Numpy array of 1d or 2d. If 2d, it is always of shape (lag, ROI)
    """
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
    acf_coeffs = acf(data, axis)
    return np.ceil( 2 * np.sum( acf_coeffs ** 2, axis = 0) ) - 2  # Always sum over the 0th axis

# z score
comparison_autocorr_zscore = comparison_data['autocorr_zscore'].ravel()
# d = data_zscore[:,0] if transpose_data else data_zscore[0]
# autocorr_zscore = acf(d)
autocorr_zscore = acf(data_zscore, axis = 0 if transpose_data else 1)[:,0]  # Doing all of it at once
print("Close after autocorrelation of zscoring?", np.allclose(autocorr_zscore if transpose_data else autocorr_zscore.T, comparison_autocorr_zscore))

# detrend
comparison_autocorr_detrend = comparison_data['autocorr_detrend'].ravel()
autocorr_detrend = acf(data_detrend, axis = 0 if transpose_data else 1)[:,0]
print("Close after autocorrelation of detrending?", np.allclose(autocorr_detrend if transpose_data else autocorr_detrend.T, comparison_autocorr_detrend))

# filter
comparison_autocorr_filter = comparison_data['autocorr_filter'].ravel()
autocorr_filter = acf(data_filter, axis = 0 if transpose_data else 1)[:,0]
print("Close after autocorrelation of filtering?", np.allclose(autocorr_filter if transpose_data else autocorr_filter.T, comparison_autocorr_filter))

# mean removal
comparison_autocorr_meanremoval = comparison_data['autocorr_meanremoval'].ravel()
autocorr_meanremoval = acf(data_meanremoval, axis = 0 if transpose_data else 1)[:,0]
print("Close after autocorrelation of mean removal?", np.allclose(autocorr_meanremoval if transpose_data else autocorr_meanremoval.T, comparison_autocorr_meanremoval))

# zscore acl
comparison_acl_zscore = comparison_data['acl_zscore'].ravel()
# acl_zscore = np.zeros(333)
# for i in range(333):
#     d = data_zscore[:,i] if transpose_data else data_zscore[i]
#     acl_zscore[i] = acl(d)
acl_zscore = acl(data_zscore, axis = 0 if transpose_data else 1)  # Doing all of it at once
print("Close acl after zscoring?", np.allclose(acl_zscore, comparison_acl_zscore))

# detrend acl
comparison_acl_detrend = comparison_data['acl_detrend'].ravel()
acl_detrend = acl(data_detrend, axis = 0 if transpose_data else 1)  # Doing all of it at once
print("Close acl after detrending?", np.allclose(acl_detrend, comparison_acl_detrend))

# filter acl
comparison_acl_filter = comparison_data['acl_filter'].ravel()
acl_filter = acl(data_filter, axis = 0 if transpose_data else 1)  # Doing all of it at once
print("Close acl after filtering?", np.allclose(acl_filter, comparison_acl_filter))

# filter mean removal
comparison_acl_meanremoval = comparison_data['acl_meanremoval'].ravel()
acl_meanremoval = acl(data_meanremoval, axis = 0 if transpose_data else 1)  # Doing all of it at once
print("Close acl after filtering?", np.allclose(acl_meanremoval, comparison_acl_meanremoval))
