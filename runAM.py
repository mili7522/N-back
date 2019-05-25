import os
import sys
import numpy as np
import pandas as pd
import utils
from runTE import computeTE


calc = utils.startCalc(measure = 'te', estimator = 'ksg')

def getLocalsForRegion(data, region_idx, param_df, compute_p = False, number_of_surrogates = 1000):
    """
    Calculates the local AM for a particular region, loading the parameters for history length and delay from file

    Arguments:
        data -- Numpy array. 2d array of shape (region, time). Preprocessing should have already been performed
        region_idx -- The index of the region under consideration, as an Int
        param_df -- 
        compute_p -- If True, computes the p value of the returned AM
    """
    
    data = data[region_idx]
    history_length, delay = param_df.iloc[region_idx] + 1  # idx starts from 0, while parameters start from 1
    if history_length > 1:
        acl = utils.acl(data)
        result, p = computeTE(k = 1, tau = 1, acl = acl, source_data = data, target_data = data, calc = calc,
                            source_history_length = history_length - 1, source_delay = delay, source_target_delay = delay + 1,
                            compute_local = True, compute_p = compute_p, number_of_surrogates = number_of_surrogates)
        return result, p
    else:
        return np.zeros_like(data), None

def getLocalsForAllRegions(data, param_df, compute_p = False):
    """
    Calculates the local AM for all regions, by calling getLocalsForRegion

    Arguments:
        data -- Numpy array of shape (region, time). Preprocessing should have already been performed
    """
    regions, timepoints = data.shape
    results = np.zeros((regions, timepoints))
    p_values = np.zeros(regions)
    for region in range(regions):
        results[region], p_values[region] = getLocalsForRegion(data, region, param_df, compute_p = compute_p)
    return results, p_values


if __name__ == "__main__":
    def test_for_one_region():
        filename = '100307.tsv'
        path = '../Data'
        region_idx = 1
        df, param_df = utils.loadData(filename, path, get_params = True, param_file = 'Results/HCP/AIS/idx/100307.csv')
        data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
        result, p = getLocalsForRegion(data, region_idx = region_idx, param_df = param_df, compute_p = False)
        utils.plotTimeseries(result)

    i = int(sys.argv[1])

    def run_individual_parameters(i, data_path, extension, save_folder, GRP = False, compute_p = True, **preprocessing_params):
        """
        Arguments:
            GRP -- True if processing the GRP data
        """
        files = utils.getAllFiles(data_path, extension)
        if GRP:
            file = files[0]
            filename = '{:02}'.format(i)    # Save the results by the subjects number
            subject_id = i
        else:
            file = files[i]
            filename = utils.basename(file)
            subject_id = None
        os.makedirs("Results/{}/AM/idx".format(save_folder), exist_ok = True)
        os.makedirs("Results/{}/AM/p_values".format(save_folder), exist_ok = True)
        
        print("Processing", i, ":", filename)
        if os.path.exists('Results/{}/AM/p_values/{}.csv'.format(save_folder, filename)):
            exit()

        param_file = 'Results/{}/AIS/idx/{}.csv'.format(save_folder, filename)
        df, param_df = utils.loadData(file, get_params = True, param_file = param_file, subject_id = subject_id)
        data = utils.preprocess(df, **preprocessing_params)
        results, p_values = getLocalsForAllRegions(data, param_df, compute_p = compute_p)
        # Add back the trimmed sections
        padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
        results = np.pad(results, padding, mode = 'constant', constant_values = 0)

        pd.DataFrame(results).to_csv('Results/{}/AM/{}_AM.csv'.format(save_folder, filename), index = None, header = None)
        pd.DataFrame(p_values).to_csv('Results/{}/AM/p_values/{}.csv'.format(save_folder, filename), index = None, header = None)
        try:
            utils.plotHeatmap(results, divergent = True)
        except:
            pass

    # HCP
    run_individual_parameters(i, data_path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP',
                                 sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25, compute_p = True)
    
    # ATX
    run_individual_parameters(i, data_path = 'Data/ATX_data', extension = '.csv', save_folder = 'ATX',
                                 sampling_rate = 1, apply_global_mean_removal = True, trim_start = 25, trim_end = 25, compute_p = True)