import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import argparse


def plotAISAcrossParams(ais_values, history_lengths, delays):
    """
    Plots a heatmap of AIS values for different parameters settings of delay and history length

    Arguments:
        ais_values -- Numpy array of AIS values for each combination of parameters. Shape = (history_lengths, delays)
        history_lengths -- Range of possible history length values
        delays -- Range of possible delay values
    """
    plt.imshow(ais_values, extent = (0.5, len(delays) + 0.5, len(history_lengths) + 0.5, 0.5))
    plt.yticks(history_lengths); plt.xticks(delays); plt.xlabel('Delay'); plt.ylabel('History Length')
    plt.colorbar(label='AIS'); plt.tight_layout()
    plt.savefig('AIS_parameters.png', dpi = 200, bbox_inches='tight')
    plt.show()

def startCalc(calc_type = 'ksg'):
    """
    Start the JIDT calculator of the selected type
    """
    # Find the location of the JIDT .jar file
    if os.path.exists("../noradrenaline/bin/infodynamics.jar"):
        jar_location = "../noradrenaline/bin/infodynamics.jar"
    else:
        jar_location = "/home/mike/Downloads/JIDT/infodynamics.jar"

    assert calc_type == 'gaussian' or calc_type == 'ksg'
    calc = utils.startCalc(measure = 'ais', estimator = calc_type, jar_location = jar_location)
    return calc

def computeAIS(k, tau, acl, data, calc, compute_local = False, compute_p = False, number_of_surrogates = 1000,
               print_surrogate = True):
    """
    Performs a calculation of Active Information Storage using JIDT

    Arguments:
        k -- History length parameter
        tau -- Delay parameter
        acl -- Autocorrelation length, used to set the dynamic correlation exclusion property
        data -- 1D Numpy array containing the time series data
        calc -- The JIDT calculator
        compute_local -- If True, the local AIS is returned instead of the average
        compute_p -- If True, the p value of the average AIS is calculated and returned
        number_of_surrogates -- Number of surrogate distributions to test to estimate the p value
    
    Returns:
        AIS -- Either locals or average
        p -- p value. Returned if compute_p is true, else None is returned
    """
    calc.setProperty("k_HISTORY", str(k))
    calc.setProperty("tau", str(tau))
    calc.setProperty('DYN_CORR_EXCL', str(acl))
    calc.setProperty( 'BIAS_CORRECTION', 'true' )  # Turns on bias correction for the gaussian estimator. The KSG Estimator is already bias adjusted
    calc.initialise()
    calc.setObservations(data)
    if compute_p:
        measDist = calc.computeSignificance(number_of_surrogates)
        if print_surrogate:
            print("Surrogate Mean:", measDist.getMeanOfDistribution())
            print("Surrogate Std:", measDist.getStdOfDistribution())
        p = measDist.pValue
    else:
        p = None
    if compute_local:
        return np.array( calc.computeLocalOfPreviousObservations() ), p
    else:
        return calc.computeAverageLocalOfObservations(), p


def getLocalsForRegion(data, calc, history_lengths, delays, region_idx = None, print_max_idx = True, compute_p = False):
    """
    Calculates the local AIS for a particular region, after first determining the appropriate values for the
    history length and delay parameters to maximise the average bias corrected AIS

    Arguments:
        data -- Numpy array. Either a 1d array containing the values for a particular region over time, or a
                2d array of shape (region, time).
                Preprocessing should have already been performed
        calc -- The JIDT calculator
        history_lengths -- Range of possible history length values
        delays -- Range of possible delay values
        region_idx -- None, or the index of the region under consideration, as an Int
        print_max_idx -- If True, prints the maximum value AIS value and the index of the corresponding parameters
                         in (history_lengths, delays)
        compute_p -- If True, computes the p value of the returned AIS

    Returns:
        result -- Numpy array of local AIS values
        ais_values -- Numpy array of shape (history_lengths, delays), giving the average AIS for each parameter combination
        max_idx -- The index of the maximum value in the array 'ais_values', given by (max_row, max_col)
        p -- p value of the computed local AIS, or None
    """
    # If the 2D (containing all the regions), then region_idx should be provided and allow the timeseries of the
    # appropriate region to be extracted
    assert data.ndim == 1 or region_idx is not None
    if region_idx is not None:
        data = data[region_idx]
    
    # Get the autocorrelation length of the timeseries data
    acl = utils.acl(data)
    print(acl)

    # Calculate bias corrected average AIS values for all combinations of history_length and delay parameters
    ais_values = np.zeros((len(history_lengths), len(delays)))
    for i, history_length in enumerate(history_lengths):
        for j, delay in enumerate(delays):                   
            ais_values[i, j], _ = computeAIS(history_length, delay, acl, data, calc, compute_local = False)

    # Find the index of the maximum bias corrected average AIS
    max_idx = utils.getMaxIdx2D(ais_values, print_ = print_max_idx)

    # Obtain the local AIS values using the parameters corresponding to the maximum bias corrected average AIS
    result, p = computeAIS(history_lengths[max_idx[0]], delays[max_idx[1]], acl, data, calc, compute_local = True,
                           compute_p = compute_p)
    return result, ais_values, max_idx, p

def getLocalsForAllRegions(data, calc, history_lengths, delays, print_max_idx = True, parameters = None, compute_p = False):
    """
    Calculates the local AIS for all regions, by calling getLocalsForRegion

    Arguments:
        data -- Numpy array of shape (region, time). Preprocessing should have already been performed
        calc -- The JIDT calculator
        history_lengths -- Range of possible history length values
        delays -- Range of possible delay values
        print_max_idx -- If True, prints the maximum value AIS value and the index of the corresponding parameters
                         in (history_lengths, delays)
        parameters --
        compute_p -- If True, computes the p value of the returned AIS
    
    Returns:
        results --
        all_max_idx --
        p_values -- 
    """
    regions, timepoints = data.shape
    results = np.zeros((regions, timepoints))
    all_max_idx = np.zeros((regions, 2))
    p_values = np.zeros(regions)
    acls = None
    for region in range(regions):
        if parameters is None:
            results[region], _, max_idx, p_values[region] = getLocalsForRegion(data, calc, history_lengths, delays, region, print_max_idx, compute_p = compute_p)
            all_max_idx[region] = np.array(max_idx)
        else:
            p = parameters[region]
            if acls is None:
                acls = utils.acl(data)  # acls is set from None to the calculated values (for all regions), so it is just run once
            results[region], p_values[region] = computeAIS(history_lengths[p[0]], delays[p[1]], acls[region], data[region], calc, compute_local = True, compute_p = compute_p)
    return results, all_max_idx, p_values


# def getAverageParametersAcrossPatients(data_path, extension, **preprocessing_params):
#     """
#     Finds the parameters for history length and delay corresponding to the maximum average AIS over the population

#     Arguments:
#         data_path -- 
#         extension -- 
#         preprocessing_params -- Includes sampling_rate / sampling_interval, apply_global_mean_removal, trim_start, trim_end
#     """
#     files = utils.getAllFiles(data_path, extension)
#     ais_values = {}
#     for k, file in enumerate(files):
#         df = utils.loadData(file)
#         data = utils.preprocess(df, **preprocessing_params)
#         regions = data.shape[0]
#         acls = None
#         for region in range(regions):
#             if region not in ais_values:
#                 ais_values[region] = np.zeros((len(files), len(history_lengths), len(delays)))
#             if acls is None:
#                 acls = utils.acl(data)
#             for i, history_length in enumerate(history_lengths):
#                 for j, delay in enumerate(delays):
#                     ais_values[region][k, i, j], _ = computeAIS(history_length, delay, acls[region], data[region], calc, compute_local = False)

#     all_max_idx = np.zeros((regions, 2))
#     for region in range(regions):
#         mean_ais_values = ais_values[region].mean(axis = 0)
#         max_idx = utils.getMaxIdx2D(mean_ais_values)
#         all_max_idx[region] = np.array(max_idx)
#     return all_max_idx


def run_individual_parameters(i, data_path, extension, save_folder, GRP = False, compute_p = True, calc_type = 'ksg',
                                 history_lengths = range(1,6), delays = range(1,2), **preprocessing_params):
    """
    Run AIS calculation for a particular set of data and parameters, selecting the parameters by maximising AIS for each subject
    
    Arguments:
        i -- The number of the file to process, as an Int
        data_path -- 
        extension -- File extension of the data
        save_folder -- 
        GRP -- True if processing the GRP data
        compute_p -- 
        calc_type -- 
        history_lengths -- 
        delays -- 
        preprocessing_params --
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
    os.makedirs("Results/{}/AIS/idx".format(save_folder), exist_ok = True)
    os.makedirs("Results/{}/AIS/p_values".format(save_folder), exist_ok = True)
    

    print("Processing file {}: {}".format(i, filename))
    if os.path.exists('Results/{}/AIS/p_values/{}.csv'.format(save_folder, filename)):
        exit()
    
    calc = startCalc(calc_type)

    df = utils.loadData(file, subject_id = subject_id)
    data = utils.preprocess(df, **preprocessing_params)
    results, all_max_idx, p_values = getLocalsForAllRegions(data, calc, history_lengths, delays, print_max_idx = False, compute_p = compute_p)
    # Add back the trimmed sections
    padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
    results = np.pad(results, padding, mode = 'constant', constant_values = 0)

    # Save results
    pd.DataFrame(results).to_csv('Results/{}/AIS/{}_AIS.csv'.format(save_folder, filename), index = None, header = None)
    
    pd.DataFrame(all_max_idx.astype(int)).to_csv('Results/{}/AIS/idx/{}.csv'.format(save_folder, filename), index = None, header = None)
    pd.DataFrame(p_values).to_csv('Results/{}/AIS/p_values/{}.csv'.format(save_folder, filename), index = None, header = None)
    try:
        utils.plotHeatmap(results, divergent = True)
    except:
        pass


#######################
#######################
    ### Population parameters
    # os.makedirs("Results/AIS Local - Population Parameters/p_values", exist_ok = True)
#    if os.path.exists('Results/Population all_max_idx.csv'):
#        all_max_idx = pd.read_csv('Results/Population all_max_idx.csv', header = None).values
#    else:
#        all_max_idx = getAverageParametersAcrossPatients(data_path = '../Data', extension = '.tsv', sampling_rate = 1.3, trim_start = 50, trim_end = 25)
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
#######################
#######################


def test_for_one_region(filename = '100307.tsv', path = '../Data', region_idx = 0,
                        history_lengths = range(1,6), delays = range(1,2),
                        calc_type = 'ksg', compute_p = False):
    calc = startCalc(calc_type)
    df = utils.loadData(filename, path)
    data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
    result, ais_values, max_idx, p_values = getLocalsForRegion(data, calc, history_lengths, delays, region_idx = region_idx, compute_p = compute_p)
    if p_values is not None:
        print("p value:", p_values)
    plotAISAcrossParams(ais_values, history_lengths, delays)
    plt.plot(result); plt.xlabel('Time'); plt.ylabel('AIS')
    plt.title("AIS Kraskov: {}[{}]\nHistory = {}, Delay = {}".format(filename, region_idx, history_lengths[max_idx[0]], delays[max_idx[1]]))


def run_experiment(experiment_number, i, local_test = False, compute_p = False, repetition = None):
    """
    Run a particular experiment with a specified set of parameters

    Arguments:
        experiment_number
        i
        local_test
        compute_p
        repetition
    """
    # Get common parameters
    if experiment_number in [0,2,3]:  # HCP experiments
        common_params = {
                         'data_path': '../Data' if local_test else 'Data/HCP',
                         'extension': '.tsv',
                         'sampling_rate': 1.3
                        }
    elif experiment_number in [1]:  # ATX experiments
        common_params = {
                         'data_path': '/media/mike/Files/Data and Results/N-back/Data/ATX_data' if local_test else 'Data/ATX_data',
                         'extension': '.csv',
                         'sampling_rate': 1
                        }
    else:
        raise Exception("No common parameters specified for experiment. Check experiment type")
    common_params['compute_p'] = compute_p
    
    def get_save_folder(folder_name):
        save_folder  = folder_name + ('_with-p' if compute_p else "")
        save_folder += ('_r{:02}'.format(repetition) if repetition is not None else "")
        return save_folder

    # Run experiment
    print("Running experiment:", experiment_number)
    if experiment_number == 0:    # HCP
        run_individual_parameters(i, save_folder = get_save_folder('HCP'), apply_global_mean_removal = True,
                                     trim_start = 50, trim_end = 25 **common_params)

    elif experiment_number == 1:  # ATX
        run_individual_parameters(i, save_folder = get_save_folder('ATX'), apply_global_mean_removal = True,
                                     trim_start = 25, trim_end = 25, **common_params)
    
    elif experiment_number == 2:  # HCP -- no global mean removal
        run_individual_parameters(i, save_folder = get_save_folder('HCP_no-mean-removal'), apply_global_mean_removal = False,
                                     trim_start = 50, trim_end = 25, **common_params)

    elif experiment_number == 3:  # HCP - using linear gaussian estimator
        run_individual_parameters(i, save_folder = get_save_folder('HCP_gaussian'), apply_global_mean_removal = True,
                                     trim_start = 50, trim_end = 25, calc_type = 'gaussian', **common_params)
    
    else:
        raise Exception("Experiment not defined")


if __name__ == "__main__":
    if os.path.exists('/home/mili7522/'):
        local_test = False
    else:
        local_test = True

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Run AIS calculation on a particular data set')
    parser.add_argument('file_number', type = int, help = 'Select the file number')
    parser.add_argument('experiment_number', type = int, help = 'Select the experiment number', default = 0)
    parser.add_argument('-p', '--compute_p', action = 'store_true', default = False)
    parser.add_argument('-r', '--repetition', type = int, default = None, help = 'Repetition')


    if len(sys.argv) > 1:
        args = parser.parse_args()
        i = args.file_number
        experiment_number = args.experiment_number
        compute_p = args.compute_p
        repetition = args.repetition

        run_experiment(experiment_number, i, local_test, compute_p, repetition)

    else:
        print("Testing for one region")
        test_for_one_region(filename = '100307.tsv', path = '../Data', region_idx = 0, calc_type = 'ksg')