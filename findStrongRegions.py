import jpype
import utils
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def getMeanAIS(ais_locals):
    #if ais_locals.shape[-1] == 405:
	if ais_locals.shape[-1] == 286:
        sum = ais_locals.sum(axis = -1)
        non_zeros = (ais_locals != 0).sum(axis = -1)
        return sum / non_zeros
    else:
        print("Last axis not time-points")
        return ais_locals

def getPValueOfAIS(ais_mean):
    '''
    Input: ais_mean is an np.array of (patients, regions)
    '''
    N = ais_mean.shape[0]
    mean = ais_mean.mean(axis = 0)
    var = ais_mean.var(axis = 0, ddof = 1)
    std = np.sqrt(var)
    t = (0 - mean) / (std / np.sqrt(N))
    # z = (0 - mean) / (std / np.sqrt(N))
    p = stats.t.cdf(t, df = (N-1))
    # p = stats.norm.cdf(z)
    return p

def countStatisticallySignificant(ind_params = True):
    if ind_params:
        p_values = utils.loadArrays(path = 'Results/ATX_data_AIS/p_values')
    else:
        p_values = utils.loadArrays(path = 'Results/ATX_data_AIS/p_values')
    p_comp = 0.05
    count = (p_values < p_comp).sum(axis = 0)
    return count

def compareCountsWithBinomial(ind_params = True):
    k = countStatisticallySignificant(ind_params)
    #n = 493  # Patients
	n = 19  # Patients
    p = 1 - stats.binom.cdf(k, n = n, p = 0.05)
    return p

# AIS from individual parameters
ais_ind = utils.loadArrays(path = 'Results/ATX_data_AIS')
ais_ind = getMeanAIS(ais_ind)
# sns.violinplot(data = ais_ind[:10].T); plt.axhline(0, color = 'black'); plt.xlabel('Regions'); plt.ylabel('Mean AIS'); plt.show()
p_ind = getPValueOfAIS(ais_ind)
#no_significant_ind = (p_ind < (0.05 / 333)).sum()
no_significant_ind = (p_ind < (0.05 / 375)).sum()
print("No. significant:", no_significant_ind)
#no_significant_ind_binomial = (compareCountsWithBinomial(ind_params = True) < (0.05 / 333)).sum()
no_significant_ind_binomial = (compareCountsWithBinomial(ind_params = True) < (0.05 / 375)).sum()
print("No. significant binomial:", no_significant_ind_binomial)

ais_ind_avg_patients = ais_ind.mean(axis = 0)
max_regions_ind = utils.getTopIdx1D(ais_ind_avg_patients)
print("Max_regions:", max_regions_ind)


# # AIS from population parameters
# ais_pop = utils.loadArrays(path = 'Results/AIS Local - Population Parameters')
# ais_pop = getMeanAIS(ais_pop)
# # sns.violinplot(data = ais_pop[:10].T); plt.axhline(0, color = 'black'); plt.xlabel('Regions'); plt.ylabel('Mean AIS'); plt.show()
# p_pop = getPValueOfAIS(ais_pop)
# no_significant_pop = (p_pop < (0.05 / 333)).sum()
# no_significant_pop_binomial = (compareCountsWithBinomial(ind_params = False) < (0.05 / 333)).sum()

# ais_pop_avg_patients = ais_pop.mean(axis = 0)
# max_regions_pop = utils.getTopIdx1D(ais_pop_avg_patients)
