import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import argparse
import time


def plotAISAcrossParams(ais_values, history_lengths, delays, show_plot = True):
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
    if show_plot:
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
               print_surrogate = False):
    """
    Performs a calculation of Active Information Storage using JIDT

    Arguments:
        k -- History length - JIDT parameter
        tau -- Delay - JIDT parameter
        acl -- Autocorrelation length, used to set the dynamic correlation exclusion property
        data -- 1D numpy array containing the time series data
        calc -- The JIDT calculator
        compute_local -- If True, a timeseries of local AIS is returned instead of the average
        compute_p -- If True, the p value of the average AIS is calculated and returned
        number_of_surrogates -- Number of surrogate distributions to test to estimate the p value
    
    Returns:
        AIS -- Either locals or average, as a float or numpy array
        p -- p value. Returned if compute_p is true, else None is returned
    """
    calc.setProperty( "k_HISTORY", str(k) )
    calc.setProperty( "tau", str(tau) )
    calc.setProperty( "DYN_CORR_EXCL", str(acl) )
    calc.setProperty( "BIAS_CORRECTION", "true" )  # Turns on bias correction for the gaussian estimator. The KSG Estimator is already bias adjusted
    calc.initialise()  # Need to initialise after properties are set
    calc.setObservations(data)
    if compute_p:
        measDist = calc.computeSignificance(number_of_surrogates)
        if print_surrogate:
            print("Surrogate Mean: {:.4f}".format( measDist.getMeanOfDistribution() ))
            print("Surrogate Std:  {:.4f}".format( measDist.getStdOfDistribution() ))
        p = measDist.pValue
    else:
        p = None
    if compute_local:
        return np.array( calc.computeLocalOfPreviousObservations() ), p
    else:
        return calc.computeAverageLocalOfObservations(), p


def runAllParameters(data, calc, history_lengths, delays, acl = None):
    """
    Calculates the bias corrected average AIS values for all combinations of history_length and delay parameters

    Arguments:
        data -- 1D numpy array containing the time series data
        calc -- The JIDT calculator
        history_lengths -- Range of possible history length values
        delays -- Range of possible delay values
        acl -- Autocorrelation length, used to set the dynamic correlation exclusion property, or None.
               If None, it is calculated from the data

    Returns:
        ais_values -- Numpy array of shape (history_lengths, delays), giving the AIS for each parameter combination
    """
    if acl is None:
        acl = utils.acl(data)
    ais_values = np.zeros((len(history_lengths), len(delays)))
    for i, history_length in enumerate(history_lengths):
        for j, delay in enumerate(delays):                   
            ais_values[i, j], _ = computeAIS(history_length, delay, acl, data, calc, compute_local = False, compute_p = False)
    return ais_values


def getLocalsForRegion(data, calc, region_idx = None, history_lengths = None, delays = None, parameters = None,
                       print_max_idx = True, compute_p = False):
    """
    Calculates the local AIS for a particular time series (data).
    The parameters of history length and delay are either provided or determined as those that maximises the 
    bias corrected average AIS value.

    Arguments:
        data -- Numpy array. Either a 1d array containing the values for a particular region over time, or a
                2d array of shape (region, time).
                Preprocessing should have already been performed
        calc -- The JIDT calculator
        history_lengths -- Range of possible history length values, or None
        delays -- Range of possible delay values, or None
        parameters -- The parameters to use for the AIS calculation, given in a list-like format as
                      (history_length, delay) or (history_length, delay, acl)
        region_idx -- The index of the region under consideration, as an Int, or None
        print_max_idx -- If True, prints the maximum average AIS value and the corresponding indices for the
                         parameters. The first value gives the maximum index in the range of possible history
                         lengths, and the second value gives the maximum index in the range of possible delays
        compute_p -- If True, computes the p value of the returned AIS

    Returns:
        result -- Numpy array of local AIS values
        ais_values -- Numpy array of shape (history_lengths, delays), giving the AIS for each parameter combination
                      None is returned instead if the parameters were provided instead of determined by maximising AIS
        parameters -- The parameters used for the AIS calculation, given as (history_length, delay, acl)
        p -- p value of the computed local AIS. Returned if compute_p is true, else None is returned
    """
    # If the 2D (containing all the regions), then region_idx should be provided and allow the timeseries of the
    # appropriate region to be extracted
    assert data.ndim == 1 or region_idx is not None
    if region_idx is not None:
        data = data[region_idx]
    
    if parameters is None:
        assert history_lengths is not None and delays is not None

        acl = utils.acl(data)  # Get the autocorrelation length of the timeseries data
        ais_values = runAllParameters(data, calc, history_lengths, delays, acl)

        # Find the index of the maximum bias corrected average AIS
        max_idx = utils.getMaxIdx2D(ais_values, print_ = print_max_idx)

        history_length = history_lengths[max_idx[0]]
        delay = delays[max_idx[1]]

        parameters = (history_length, delay, acl)
    else:
        # If the autocorrelation length of the data was not provided, then calculate it
        if len(parameters) == 2:
            acl = utils.acl(data)
            history_length, delay = parameters
            parameters = (history_length, delay, acl)
        else:
            history_length, delay, acl = parameters
        ais_values = None
    
    # Obtain the local AIS values using the parameters corresponding to the maximum bias corrected average AIS
    result, p = computeAIS(history_length, delay, acl, data, calc, compute_local = True, compute_p = compute_p)
    return result, ais_values, parameters, p


def getLocalsForAllRegions(data, calc, history_lengths = None, delays = None, parameters = None, print_max_idx = True, compute_p = False):
    """
    Calculates the local AIS for all regions, by calling getLocalsForRegion

    Arguments:
        data -- Numpy array of shape (region, time). Preprocessing should have already been performed
        calc -- The JIDT calculator
        history_lengths -- Range of possible history length values, or None
        delays -- Range of possible delay values, or None
        print_max_idx -- If True, prints the maximum average AIS value and the corresponding indices for the
                         parameters. The first value gives the maximum index in the range of possible history
                         lengths, and the second value gives the maximum index in the range of possible delays
        parameters -- A DataFrame or numpy array containing a column of history lengths and a column of delays
                      Each row should correspond to a particular region
        compute_p -- If True, computes the p value of the returned AIS
    
    Returns:
        results -- A numpy array of shape (regions, timepoints), containing the local AIS values for each region
        all_parameters -- A numpy array with three columns, containing the (history_length, delay, acl) of each region
        p_values -- A numpy array of all returned p values (or Nones if compute_p is False). Each row corresponds to a region
    """
    regions, timepoints = data.shape

    # Initialise
    results = np.zeros((regions, timepoints))
    all_parameters = np.zeros((regions, 3), dtype = int)
    p_values = np.zeros(regions)

    for region in range(regions):
        # Either parameters are provided, or the range of possible history lengths and delays should be provided
        if parameters is None:
            assert history_lengths is not None and delays is not None
            params = None
        else:
            if isinstance(parameters, pd.DataFrame):
                params = parameters.loc[region].values
            else:  # Numpy array or list, etc
                params = parameters[region]
        results[region], _, params, p_values[region] = getLocalsForRegion(data, calc, region, history_lengths, delays, params,
                                                                          print_max_idx, compute_p)
        all_parameters[region] = np.array(params)
        utils.update_progress(region/regions)  # Print progress bar
    return results, all_parameters, p_values


def getPopulationParameters(data_path, extension, calc, history_lengths, delays, **preprocessing_params):
    """
    Finds the parameters for history length and delay by considering the AIS values which have been averaged over the population
    AIS is calculated for each subject, for each combination of parameters and then averaged across the subjects
    The history length and delay which gives the maximum averaged AIS is chosen as the population parameters

    Arguments:
        data_path -- Location of the data files
        extension -- File extension of the data
        calc -- The JIDT calculator
        history_lengths -- Range of possible history length values
        delays -- Range of possible delay values
        preprocessing_params -- Parameters passed to utils.preprocess for preprocessing the time series data.
                                Includes sampling_rate / sampling_interval, apply_global_mean_removal, trim_start, trim_end
    
    Returns:
        parameters -- A numpy array with two columns, containing the (history_length, delay) of each region
    """
    print("Finding population parameters")
    files = utils.getAllFiles(data_path, extension)

    data_all = utils.loadData(files[0], return_all_subjects = True)

    if data_all.ndim == 3:
        # Check to see if it's the GRP data, which will be returned as a 3D array of (regions, time points, subjects)
        number_of_subjects = data_all.shape[2]
    else:
        number_of_subjects = len(files)

    regions = data_all.shape[0]
    # Initialise
    ais_values = np.empty((regions, number_of_subjects, len(history_lengths), len(delays)))

    for i in range(number_of_subjects):
        if data_all.ndim == 3:
            data = data_all[:,:,i]  # If it's the GRP data, get the data for each subject by slicing on the 3rd dimension
        elif i == 0:
            data = data_all
        else:  # If it's not the GRP data, just load the next file when the first one is done
            data = utils.loadData(files[i])

        data = utils.preprocess(data, **preprocessing_params)
        acls = None  # Just want to calculate the acls once for all the regions, so assign it as None first, and then replace it
        for region in range(regions):
            if acls is None:
                acls = utils.acl(data)
            ais_values[region, i] = runAllParameters(data[region], calc, history_lengths, delays, acls[region])
            
            # Print progress bar
            utils.update_progress((i * regions + region) / (number_of_subjects * regions),
                                  end_text = "  Subject {} of {}".format(i + 1, number_of_subjects))

    parameters = np.zeros((regions, 2), dtype = int)
    mean_ais_values = ais_values.mean(axis = 1)  # Average over the subjects

    for region in range(regions):
        max_idx = utils.getMaxIdx2D(mean_ais_values[region])  # Find the argmax across the averaged AIS values
        history_length = history_lengths[max_idx[0]]
        delay = delays[max_idx[1]]
        parameters[region] = (history_length, delay)

    return parameters


def run(i, data_path, extension, save_folder, GRP = False, compute_p = True, calc_type = 'ksg',
        use_population_parameters = False, history_lengths = range(1,6), delays = range(1,2),
        **preprocessing_params):
    """
    Run AIS calculation for a particular subject, using parameters which are either selected by 
    maximising the bias corrected average AIS for the individual subject, or parameters which are
    first determined by averaging the AIS across the population
    
    Arguments:
        i -- An Int which states which file or subject to load and process
        data_path -- Location of the data files
        extension -- File extension of the data (eg. .csv, .tsv, .mat)
        save_folder -- Subfolder of the 'Results' directory in which to save the local AIS values, parameters and p_values
        GRP -- Set to True if processing the GRP data, which is one array of dimension (region, timepoints, subject)
        compute_p -- If True, computes the p value of the returned AIS
        calc_type -- The type of estimator to use for the JIDT calculator - 'gaussian' or 'ksg'
        history_lengths -- Range of possible history length values
        delays -- Range of possible delay values
        preprocessing_params -- Parameters passed to utils.preprocess for preprocessing the time series data.
                                Includes sampling_rate / sampling_interval, apply_global_mean_removal, trim_start, trim_end
    """
    start_time = time.time()
    files = utils.getAllFiles(data_path, extension)
    if GRP:
        file = files[0]
        filename = '{:02}'.format(i)    # Save the results by the subjects number
        subject_id = i
    else:
        file = files[i]
        filename = utils.basename(file)
        subject_id = None
    # Makes folders to save the results
    os.makedirs("Results/{}/AIS/params".format(save_folder), exist_ok = True)
    os.makedirs("Results/{}/AIS/p_values".format(save_folder), exist_ok = True)

    print("Processing file {}: {}".format(i, filename))
    if os.path.exists('Results/{}/AIS/p_values/{}_p.csv'.format(save_folder, filename)):
        print("Result already present")
        exit()  # If the results have already been saved, avoid running again
    
    calc = startCalc(calc_type)

    # Load and preprocess data
    df = utils.loadData(file, subject_id = subject_id)
    data = utils.preprocess(df, **preprocessing_params)

    if use_population_parameters:
        # If using population parameters, either load the pre-calculated parameters, or calculate them and save to file
        if os.path.exists('Results/{}/AIS/population_params.csv'.format(save_folder)):
            parameters = pd.read_csv('Results/{}/AIS/population_params.csv'.format(save_folder))
        else:
            parameters = getPopulationParameters(data_path, extension, calc, history_lengths, delays, **preprocessing_params)
            pd.DataFrame(parameters, columns = ['k', 'tau']).to_csv('Results/{}/AIS/population_params.csv'.format(save_folder), index = None)
    else:
        parameters = None

    results, all_parameters, p_values = getLocalsForAllRegions(data, calc, history_lengths, delays, parameters,
                                                               print_max_idx = False, compute_p = compute_p)

    # Add back the trimmed sections at the start and end of the timeseries by padding with zeros
    padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
    results = np.pad(results, padding, mode = 'constant', constant_values = 0)

    # Save results
    pd.DataFrame(results).to_csv('Results/{}/AIS/{}_AIS.csv'.format(save_folder, filename), index = None, header = None)
    params_df = pd.DataFrame(all_parameters, columns = ['k', 'tau', 'acl'])
    params_df.to_csv('Results/{}/AIS/params/{}_params.csv'.format(save_folder, filename), index = None)
    pd.DataFrame(p_values).to_csv('Results/{}/AIS/p_values/{}_p.csv'.format(save_folder, filename), index = None, header = None)

    print("\nTime taken:", round((time.time() - start_time) / 60, 2), 'min')
    try:
        utils.plotHeatmap(results, divergent = True)
    except:
        pass


def test_for_one_region(filename = '100307.tsv', path = '../Data', region_idx = 0,
                        history_lengths = range(1,6), delays = range(1,2),
                        calc_type = 'ksg', compute_p = False):
    calc = startCalc(calc_type)
    df = utils.loadData(filename, path)
    data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
    result, ais_values, parameters, p_values = getLocalsForRegion(data, calc, region_idx, history_lengths, delays, compute_p = compute_p)
    if p_values is not None:
        print("p value:", p_values)
    plotAISAcrossParams(ais_values, history_lengths, delays, show_plot = False)
    plt.figure()
    plt.plot(result); plt.xlabel('Time'); plt.ylabel('AIS')
    plt.title("AIS Kraskov: {}[{}]\nHistory = {}, Delay = {}".format(filename, region_idx, parameters[0], parameters[1]))
    plt.show()


def run_experiment(experiment_number, i, local_test = False, compute_p = False, repetition = None):
    """
    Run a particular experiment with a specified set of parameters and data
    A folder is created for the results. The folder name is modified by the repetition number,
    and whether p values are calculated

    Arguments:
        experiment_number -- An INt which states which experiment to run
        i -- An Int which states which file or subject to load and process
        local_test -- If True, set file paths for local testing
        compute_p -- If True, computes the p value of the returned AIS
        repetition -- None, or an Int which specifies the repetition number of the run
                      Repetitions are saved in their own folder with the number as a suffix
    """
    # Get parameters which are common across a particular experiment type
    if experiment_number in [0,2,3,6]:  # HCP experiments
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
        """
        Modifies the folder name to include an indication if p values are calculated, and
        adds a suffix indicating the repetition number if repetition != None
        """
        save_folder  = folder_name + ('_with-p' if compute_p else "")
        save_folder += ('_r{:02}'.format(repetition) if repetition is not None else "")
        return save_folder

    # Run experiment
    print("Running experiment:", experiment_number)
    if experiment_number == 0:    # HCP
        run(i, save_folder = get_save_folder('HCP'), apply_global_mean_removal = True,
            trim_start = 50, trim_end = 25, **common_params)

    elif experiment_number == 1:  # ATX
        run(i, save_folder = get_save_folder('ATX'), apply_global_mean_removal = True,
            trim_start = 25, trim_end = 25, **common_params)
    
    elif experiment_number == 2:  # HCP -- no global mean removal
        run(i, save_folder = get_save_folder('HCP_no-mean-removal'), apply_global_mean_removal = False,
            trim_start = 50, trim_end = 25, **common_params)

    elif experiment_number == 3:  # HCP - using linear gaussian estimator
        run(i, save_folder = get_save_folder('HCP_gaussian'), apply_global_mean_removal = True,
            trim_start = 50, trim_end = 25, calc_type = 'gaussian', **common_params)
    
    elif experiment_number == 6:  # HCP - using population parameters
        run(i, save_folder = get_save_folder('HCP_pop-param'), apply_global_mean_removal = True,
            trim_start = 50, trim_end = 25, use_population_parameters = True, **common_params)

    else:
        raise Exception("Experiment not defined")



############################################################################################################
if __name__ == "__main__":
    if os.path.exists('/home/mili7522/'):
        local_test = False
    else:
        local_test = True


    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Run AIS calculation on a particular data set.'
                                                    + ' Input two integers, the subject number followed by the experiment number.')
    parser.add_argument('subject_number', type = int, help = 'the number of the subject or file to load')
    parser.add_argument('experiment_number', type = int, help = 'the number of the experiment to perform', default = 0)
    parser.add_argument('-p', '--compute_p', action = 'store_true', default = False, help = 'set as True to calculate p-values (with 1000 surrogates)')
    parser.add_argument('-r', '--repetition', metavar = 'R', type = int, default = None, help = 'repetition number. Default = None')


    if len(sys.argv) > 1:
        args = parser.parse_args()
        subject_number = args.subject_number
        experiment_number = args.experiment_number
        compute_p = args.compute_p
        repetition = args.repetition

        run_experiment(experiment_number, subject_number, local_test, compute_p, repetition)

    else:
        print("Testing for one region")
        test_for_one_region(filename = '100307.tsv', path = '../Data', region_idx = 0, calc_type = 'ksg')