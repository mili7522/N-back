import jpype
import utils
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

history_lengths = range(1, 6)
delays = range(1, 2)

calc = utils.startCalc(measure = 'ais', estimator = 'ksg')

def plotAISAcrossParams(ais_values):
    plt.imshow(ais_values, extent = (0.5, len(delays)+0.5, len(history_lengths)+0.5, 0.5))
    plt.yticks(history_lengths); plt.xticks(delays); plt.xlabel('Delay'); plt.ylabel('History Length')
    plt.colorbar(label='AIS'); plt.tight_layout()
    plt.savefig('AIS_parameters.png', dpi = 200, bbox_inches='tight')
    plt.show()

def computeAIS(k, tau, acl, data, calc, compute_local = False, compute_p = False, number_of_surrogates = 1000):
    """
    Performs a calculation of Active Information Storage using JIDT

    Arguments:
        k -- History length parameter
        tau -- Delay parameter
        acl -- Auto correlation length, used to set the dynamic correlation exclusion property
        data -- 1D Numpy array
        calc -- The JIDT calculator
        compute_local -- If True, the local AIS is returned instead of the expectation
        compute_p  -- If True, the p value of the average AIS
        number_of_surrogates
    
    Returns:
        AIS -- Either locals or average
        p -- p value. Returned if compute_p is true
    """
    calc.setProperty("k_HISTORY", str(k))
    calc.setProperty("tau", str(tau))
    calc.setProperty('DYN_CORR_EXCL', str(acl))
    calc.setObservations(data)
    if compute_p:
        measDist = calc.computeSignificance(number_of_surrogates)
        # print("Surrogate Mean:", measDist.getMeanOfDistribution())
        # print("Surrogate Std:", measDist.getStdOfDistribution())
        p = measDist.pValue
    else:
        p = None
    if compute_local:
        return np.array( calc.computeLocalOfPreviousObservations() ), p
    else:
        return calc.computeAverageLocalOfObservations(), p  # KSG Estimator is already bias adjusted


def getLocalsForRegion(data, region_idx = None, print_max_idx = True, compute_p = False):
    """
    Calculates the local AIS for a particular region, after first determining the appropriate values for the history length and delay parameters

    Arguments:
        data -- Numpy array. Either a 1d array containing the values for a particular region over time, or a 2d array of shape (region, time)
                Preprocessing should have already been performed
        region_idx -- None, or the index of the region under consideration, as an Int
        print_max_idx -- If True, prints the maximum value AIS value and the index of the corresponding parameters in (history_lengths, delays)
        compute_p -- If True, computes the p value of the returned AIS
    """
    assert data.ndim == 1 or region_idx is not None
    if region_idx is not None:
        data = data[region_idx]
    
    acl = utils.acl(data)

    ais_values = np.zeros((len(history_lengths), len(delays)))
    
    for i, history_length in enumerate(history_lengths):
        for j, delay in enumerate(delays):                   
            ais_values[i, j], _ = computeAIS(history_length, delay, acl, data, calc, compute_local = False)

    max_idx = utils.getMaxIdx2D(ais_values, print_ = print_max_idx)

    result, p = computeAIS(history_lengths[max_idx[0]], delays[max_idx[1]], acl, data, calc, compute_local = True, compute_p = compute_p)
    return result, ais_values, max_idx, p

def getLocalsForAllRegions(data, print_max_idx = True, parameters = None, compute_p = False):
    """
    Calculates the local AIS for all regions, by calling getLocalsForRegion

    Arguments:
        data -- Numpy array of shape (region, time). Preprocessing should have already been performed
    """
    regions, timepoints = data.shape
    results = np.zeros((regions, timepoints))
    all_max_idx = np.zeros((regions, 2))
    p_values = np.zeros(regions)
    acls = None
    for region in range(regions):
        if parameters is None:
            results[region], _, max_idx, p_values[region] = getLocalsForRegion(data, region, print_max_idx, compute_p = compute_p)
            all_max_idx[region] = np.array(max_idx)
        else:
            p = parameters[region]
            if acls is None:
                acls = utils.acl(data)
            results[region], p_values[region] = computeAIS(history_lengths[p[0]], delays[p[1]], acls[region], data[region], calc, compute_local = True, compute_p = compute_p)
    return results, all_max_idx, p_values

def getAverageParametersAcrossPatients(path, extension, **preprocessing_params):
    """
    Finds the parameters for history length and delay corresponding to the maximum average AIS over the population

    Arguments:
        path -- 
        extension -- 
        preprocessing_params -- Includes sampling_rate / sampling_interval, apply_global_mean_removal, trim_start, trim_end
    """
    files = utils.getAllFiles(path, extension)
    ais_values = {}
    for k, file in enumerate(files):
        df = utils.loadData(file)
        data = utils.preprocess(df, **preprocessing_params)
        regions = data.shape[0]
        acls = None
        for region in range(regions):
            if region not in ais_values:
                ais_values[region] = np.zeros((len(files), len(history_lengths), len(delays)))
            if acls is None:
                acls = utils.acl(data)
            for i, history_length in enumerate(history_lengths):
                for j, delay in enumerate(delays):
                    ais_values[region][k, i, j], _ = computeAIS(history_length, delay, acls[region], data[region], calc, compute_local = False)

    all_max_idx = np.zeros((regions, 2))
    for region in range(regions):
        mean_ais_values = ais_values[region].mean(axis = 0)
        max_idx = utils.getMaxIdx2D(mean_ais_values)
        all_max_idx[region] = np.array(max_idx)
    return all_max_idx



if __name__ == "__main__":
    def test_for_one_region():
        filename = '100307.tsv'
        path = '../Data'
        region_idx = 0
        df = utils.loadData(filename, path)
        data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
        result, ais_values, max_idx, p_value = getLocalsForRegion(data, region_idx = region_idx, compute_p = False)
        plotAISAcrossParams(ais_values)
        plt.plot(result); plt.xlabel('Time'); plt.ylabel('AIS')
        plt.title("AIS Kraskov: {}[{}]\nHistory = {}, Delay = {}".format(filename, region_idx, history_lengths[max_idx[0]], delays[max_idx[1]]))

    i = int(sys.argv[1])

    def run_individual_parameters(i, path, extension, save_folder, GRP = False, **preprocessing_params):
        """
        Arguments:
            GRP -- True if processing the GRP data
        """
        files = utils.getAllFiles(path, extension)
        if GRP:
            file = files[0]
        else:
            file = files[i]
        os.makedirs("Results/{}/AIS/idx".format(save_folder), exist_ok = True)
        os.makedirs("Results/{}/AIS/p_values".format(save_folder), exist_ok = True)
        
        print("Processing", i, ":", file)
        if os.path.exists('Results/{}/AIS/p_values/{}.csv'.format(save_folder, utils.basename(file))):
            exit()
        if GRP:
            df = utils.loadData(file, subject_id = i)
        else:
            df = utils.loadData(file)
        data = utils.preprocess(df, **preprocessing_params)
        results, all_max_idx, p_values = getLocalsForAllRegions(data, print_max_idx = False, compute_p = True)
        # Add back the trimmed sections
        padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
        results = np.pad(results, padding, mode = 'constant', constant_values = 0)

        if GRP:
            file = '{:02}'.format(i)    # Save the results by the subjects number
        pd.DataFrame(results).to_csv('Results/{}/AIS/{}_AIS.csv'.format(save_folder, utils.basename(file)), index = None, header = None)
        pd.DataFrame(all_max_idx.astype(int)).to_csv('Results/{}/AIS/idx/{}.csv'.format(save_folder, utils.basename(file)), index = None, header = None)
        pd.DataFrame(p_values).to_csv('Results/{}/AIS/p_values/{}.csv'.format(save_folder, utils.basename(file)), index = None, header = None)
        utils.plotHeatmap(results, divergent = True)

    # HCP
    run_individual_parameters(i, path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP',
                                sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)


    ### Population parameters
    # os.makedirs("Results/AIS Local - Population Parameters/p_values", exist_ok = True)
#    if os.path.exists('Results/Population all_max_idx.csv'):
#        all_max_idx = pd.read_csv('Results/Population all_max_idx.csv', header = None).values
#    else:
#        all_max_idx = getAverageParametersAcrossPatients(path = '../Data', extension = '.tsv', sampling_rate = 1.3, trim_start = 50, trim_end = 25)
#        pd.DataFrame(all_max_idx.astype(int)).to_csv('Results/Population all_max_idx.csv', index = None, header = None)
#    # for i, file in enumerate(utils.getAllFiles()):
#    # if os.path.exists('Results/AIS Local - Population Parameters/{}_AIS.csv'.format(file.split('.')[0])):
#    #     continue
#    print("Processing", i, ":", file)
#    df = utils.loadData(file)
#    results, _, p_values = getLocalsForAllRegions(df, print_max_idx = False, parameters = all_max_idx, compute_p = True)
#    pd.DataFrame(results).to_csv('Results/AIS Local - Population Parameters/{}_AIS.csv'.format(utils.basename(file)), index = None, header = None)
#    pd.DataFrame(p_values).to_csv('Results/AIS Local - Population Parameters/p_values/{}.csv'.format(utils.basename(file)), index = None, header = None)
#    # utils.plotHeatmap(results, divergent = True)


