import glob
import os
import sys
import numpy as np
import pandas as pd
import utils
import argparse
import time


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
    calc = utils.startCalc(measure = 'te', estimator = calc_type, jar_location = jar_location)
    return calc


def computeTE(k, tau, acl, source_data, target_data, calc, source_history_length = 1,
              source_delay = 1, source_target_delay = 1, compute_local = False,
              compute_p = False, number_of_surrogates = 100, print_surrogate = True):
    """
    Performs a calculation of Transfer Entropy using JIDT

    Arguments:
        k -- History length of target - JIDT parameter
        tau -- Delay between target time points - JIDT parameter
        acl -- Auto correlation length, used to set the dynamic correlation exclusion property
        source_data -- 1D Numpy array containing the time series of the source region
        target_data -- 1D Numpy array containing the time series of the target region
        calc -- The JIDT calculator
        source_history_length -- History length of source - JIDT parameter
        source_delay -- Delay of source time points - JIDT parameter
        source_target_delay -- Delay between source to target - JIDT parameter
        compute_local -- If True, a timeseries of the local TE is returned instead of the average
        compute_p  -- If True, the p value of the average TE is calculated and returned
        number_of_surrogates -- Number of surrogate distributions to test to estimate the p value
    
    Returns:
        TE -- Either locals or average, as a float or numpy array
        p -- p value. Returned if compute_p is true, else None is returned
    """
    calc.setProperty( "DYN_CORR_EXCL", str(acl) )
    calc.setProperty( "BIAS_CORRECTION", "true" )  # Turns on bias correction for the gaussian estimator. The KSG Estimator is already bias adjusted
    calc.initialise(k, tau, source_history_length, source_delay, source_target_delay)  # Need to initialise after properties are set
    calc.setObservations(source_data, target_data)
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


def getLocalsForRegionPair(data, source_idx, target_idx, param_df, calc, compute_p = False):
    """
    Calculates the local TE for a particular source region - target region pair
    The parameters for target history length and target delay are passed in by param_df

    Arguments:
        data -- Numpy array. 2d array of shape (region, time). Preprocessing should have already been performed
        source_idx -- The index of the region under consideration, as an Int
        target_idx -- The index of the region under consideration, as an Int
        param_df -- Pandas DataFrame containing the parameters used for each region, with the columns 'k', 'tau' and 'acl'
                    (acl is not present if the experiment has set_k_to_0 = True)
        calc -- The JIDT calculator
        compute_p -- If True, computes the p value of the returned TE
    
    Returns:
        result -- Numpy array of local TE values
        p -- p value of the computed local TE. Returned if compute_p is true, else None is returned
        acl -- The auto-correlation length used to set the dynamic correlation exclusion property for
               the TE calculation
    """
    # Extract the source and target time series from the full data array
    source_data = data[source_idx]
    target_data = data[target_idx]
    
    # Extract the history length and delay parameters from param_df
    history_length, delay = param_df.loc[target_idx, ['k', 'tau']]

    if 'acl' in param_df:
        acl = max(param_df.loc[source_idx, 'acl'], param_df.loc[target_idx, 'acl'])  # Take the larger value from the source and target
    else:
        acl = max(utils.acl(source_data), utils.acl(target_data))  # Calculate the acl if it's not present in param_df

    result, p = computeTE(history_length, delay, acl, source_data, target_data, calc, compute_local = True, compute_p = compute_p)
    return result, p, acl


def getLocalsForAllRegionPairs(data, param_df, calc, compute_p = False, print_every = 50, save_every = 20, saver = None,
                               results = None, p_values = None, idx_values = None):
    """
    Calculates the local TE for all pairs of regions, by calling getLocalsForRegionPair

    Arguments:
        data -- Numpy array of shape (region, time). Preprocessing should have already been performed
        param_df -- Pandas DataFrame containing the parameters used for each region, in the columns 'k', 'tau' and 'acl'
                    (acl is not present if the experiment has set_k_to_0 = True)
        calc -- The JIDT calculator
        compute_p -- If True, computes the p value of the returned TE
        print_every -- None, or Int giving the number of regions to calculate before printing an update of the progress
        save_every -- None, or Int giving the number of regions to calculate before saving the current state of the results
        saver -- TEResultSaver object, used to save the intermediate results
        results -- Loaded results from previous run, or None
        p_values -- Loaded p_values from previous run, or None
        idx_values -- Tuple of the last (source_idx, target_idx) of the saved results from previous run, or None
    
    Returns:
        results -- A numpy array of shape (regions, regions, timepoints), containing the local TE values for each region to region pair
                   The first dimension corresponds to the source region, the second dimension corresponds to the target region
        p_values -- A numpy array of shape (region, region) containing all returned p values (or Nones if compute_p is False).
                    Each row corresponds to a source region, and each column corresponds to a target region
    """
    regions, timepoints = data.shape

    if idx_values is None:
        # Start from the beginning. Initialise
        starting_source_idx, starting_target_idx = 0, 0
        results = np.zeros((regions, regions, timepoints))
        p_values = np.zeros((regions, regions))
    else:
        # Continue from where the loaded results left off
        starting_source_idx, starting_target_idx = idx_values
        assert results is not None and p_values is not None

    # Calculate the local TE for all source / target pairs
    for source_idx in range(starting_source_idx, regions):
        for target_idx in range(regions):
            if source_idx == starting_source_idx and target_idx < starting_target_idx:
                continue  # Start calculations at (starting_source_idx, starting_target_idx)
            if source_idx == target_idx:
                results[source_idx, target_idx] = np.nan  # Don't include the diagonal in any mean calculations
            else:
                if print_every is not None and (target_idx % print_every == 0) :
                    # Print progress bar
                    utils.update_progress((source_idx * regions + target_idx) / regions ** 2,
                                          end_text = "  {:4} -> {:4}".format(source_idx, target_idx))

                results[source_idx, target_idx], p_values[source_idx, target_idx], _ = getLocalsForRegionPair(data, source_idx, target_idx,
                                                                                                              param_df, calc, compute_p)
                
                # Save intermediate results
                if save_every is not None and saver is not None and (target_idx % save_every == 0):
                    saver.save_intermediate_result(results, p_values, (source_idx, target_idx + 1))  # If loaded, start from (source_idx, target_idx + 1)

    return results, p_values


class TEResultSaver:
    def __init__(self, filename, save_folder, raw_save_root = "/scratch/InfoDynFuncStruct/Mike/N-back/"):
        """
        Arguments:
            filename -- Base name of the files to be saved
            save_folder -- Folder to save the final results (TE in and out of each region)
            raw_save_root -- Root folder to save the raw TE results
        """
        self.filename = filename
        self.save_folder = save_folder
        self.raw_save_root = raw_save_root
        
        os.makedirs(os.path.join(raw_save_root, "Results/{}/TE/raw/p_values".format(save_folder)), exist_ok = True)
        os.makedirs("Results/{}/TE/In-Target".format(save_folder), exist_ok = True)
        os.makedirs("Results/{}/TE/Out-Source".format(save_folder), exist_ok = True)
    
    def save_intermediate_result(self, results, p_values, idx_values):
        """
        Arguments:
            idx_values -- Tuple of (source_idx, target_idx) for the next source and target idx to be processed
        """
        self.save_raw(results, p_values, compress = False)  # Don't compress the intermediate files, for faster saving and loading
        # Write the values of the next (source_idx, target_idx) in a file
        # The presence of this file will indicate that the final results have not be reached. It is deleted after the final results are saved
        with open(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(self.save_folder, self.filename)), 'w') as f:
            f.write(str(idx_values[0]) + "," + str(idx_values[1]))

    def save_final_result(self, results, p_values, padding = ((0,0), (0,0)), compress = False):
        """
        Saves the TE results - the raw array of shape (source_region, target_region, time) as well as the averaged local TE
        out of each source region and into each target region

        Arguments:
            results -- Numpy array containing the raw TE results, of shape (source_region, target_region, time)
            p_values -- Numpy array containing the p values, of shape (source_region, target_region)
            padding -- Tuple of tuples containing the number of spaces to pad the TE results at the start and end of each dimension
                       Follows the requirement specified by np.pad
            compress -- If True, save the raw results as a npz format instead of npy
        """
        self.save_raw(results, p_values, compress)

        # Take the average across source / target regions, ignoring the diagonals where source = target
        target_te = np.nanmean(results, axis = 0)  # Average across all sources
        source_te = np.nanmean(results, axis = 1)  # Average across all targets

        # Add back the trimmed sections at the start and end of the timeseries by padding with zeros
        target_te = np.pad(target_te, padding, mode = 'constant', constant_values = 0)
        source_te = np.pad(source_te, padding, mode = 'constant', constant_values = 0)

        pd.DataFrame(target_te).to_csv('Results/{}/TE/In-Target/{}.csv'.format(self.save_folder, self.filename), index = None, header = None)
        pd.DataFrame(source_te).to_csv('Results/{}/TE/Out-Source/{}.csv'.format(self.save_folder, self.filename), index = None, header = None)
        
        # Clean up intermediate save files
        if compress:  # The .npz files are kept, and npy files are removed
            os.remove(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}.npy".format(self.save_folder, self.filename)))
            os.remove(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_p.npy".format(self.save_folder, self.filename)))
        os.remove(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(self.save_folder, self.filename)))
        # The file containing the next (source_idx, target_idx) is deleted after the final results are saved

    def save_raw(self, results, p_values, compress):
        """
        Saves the raw files

        Arguments:
            results -- Numpy array containing the raw TE results, of shape (source_region, target_region, time)
            p_values -- Numpy array containing the p values, of shape (source_region, target_region)
            compress -- If True, save the raw results as a npz format instead of npy. Only used when saving the final result
        """
        if compress:
            np.savez_compressed(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}.npz".format(self.save_folder, self.filename)), results = results)
            np.savez_compressed(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_p.npz".format(self.save_folder, self.filename)), p_values = p_values)
        else:
            # Save temp file first then rename, in case the process gets killed during the save
            np.save(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_temp.npy".format(self.save_folder, self.filename)), results)
            np.save(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_temp.npy".format(self.save_folder, self.filename)), p_values)
            os.replace(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_temp.npy".format(self.save_folder, self.filename)),
                       os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}.npy".format(self.save_folder, self.filename)))
            os.replace(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_temp.npy".format(self.save_folder, self.filename)),
                       os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_p.npy".format(self.save_folder, self.filename)))


def test_for_one_pair(filename = '100307.tsv', path = '../Data', source_region = 1, target_region = 0,
                      param_file = 'Results/HCP/AIS/params/100307_params.csv', calc_type = 'ksg', compute_p = False):
    calc = startCalc(calc_type)
    df, param_df = utils.loadData(filename, path, get_params = True, param_file = param_file)
    data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
    result, p_values, acl = getLocalsForRegionPair(data, source_region, target_region, param_df, calc, compute_p = compute_p)
    if p_values is not None:
        print("p value:", p_values)
    print('Auto-correlation length:', acl)
    utils.plotTimeseries(result)


def run(i, data_path, extension, save_folder, raw_save_root = "/scratch/InfoDynFuncStruct/Mike/N-back/", save_every = 20,
        GRP = False, compute_p = True, compress = False, set_k_to_0 = False, calc_type = 'ksg', **preprocessing_params):
    """
    Run TE calculation for a particular subject. Parameters are loaded from file, based on the AIS calculation, or set
    to 0 if set_k_to_0 is True
    
    Arguments:
        i -- An Int which states which file or subject to load and process
        data_path -- Location of the data files
        extension -- File extension of the data (eg. .csv, .tsv, .mat)
        save_folder -- Subfolder of the 'Results' directory in which to save the local AIS values, parameters and p_values
        raw_save_root -- Location to save the raw local TE values (as a npz or npy file)
        save_every -- None, or Int giving the number of regions to calculate before saving the current state of the results
        GRP -- Set to True if processing the GRP data, which is one array of dimension (region, timepoints, subject)
        compute_p -- If True, computes the p value of the returned AIS
        calc_type -- The type of estimator to use for the JIDT calculator - 'gaussian' or 'ksg'
        compress -- If True, the raw TE values are saved as a compressed npz file instead of an npy file
        set_k_to_0 -- If True, skip loading of k and l parameters, instead initialising the DataFrame to zeros
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
    
    print("Processing file {}: {}".format(i, filename))
    # Check for the presence of the current_idx file
    # If it's not present, then either no calculations have been done, or the final results have already been saved
    if os.path.isfile(os.path.join(raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(save_folder, filename))):
        # Load previous results, which are always saved in the uncompressed format
        results = np.load(os.path.join(raw_save_root, "Results/{}/TE/raw/{}.npy".format(save_folder, filename)))
        p_values = np.load(os.path.join(raw_save_root, "Results/{}/TE/raw/p_values/{}_p.npy".format(save_folder, filename)))
        with open(os.path.join(raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(save_folder, filename)), 'r') as f:
            idx_values = f.readline()
        idx_values = list(map(int,idx_values.split(',')))
        print("Loading previous results")
        print("Starting from index", idx_values)
    else:
        results, p_values, idx_values = None, None, None
        # Check both compressed and uncompressed options. If this file exists but the current_idx file doesn't then the
        # final results have already been saved. Exit to avoid running again
        if glob.glob(os.path.join(raw_save_root, "Results/{}/TE/raw/p_values/{}_p.np*".format(save_folder, filename))):
            print("Result already present")
            exit()

    # Load parameter file
    param_file = 'Results/{}/AIS/params/{}_params.csv'.format(save_folder, filename)
    if set_k_to_0:
        df = utils.loadData(file, get_params = False, subject_id = subject_id)
        param_df = pd.DataFrame( np.zeros((len(df), 2), dtype = int), columns = ['k', 'tau'])
    else:
        df, param_df = utils.loadData(file, get_params = True, param_file = param_file, subject_id = subject_id)

    data = utils.preprocess(df, **preprocessing_params)
    saver = TEResultSaver(filename, save_folder, raw_save_root)
    calc = startCalc(calc_type)

    # Do the calculations
    results, p_values = getLocalsForAllRegionPairs(data, param_df, calc, compute_p, saver = saver, 
                                                   save_every = save_every, results = results,
                                                   p_values = p_values, idx_values = idx_values)

    # Save the final results
    # Add back the trimmed sections at the start and end of the timeseries by padding with zeros
    padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
    saver.save_final_result(results, p_values, padding = padding, compress = compress)
    print("\nTime taken:", round((time.time() - start_time) / 60, 2), 'min')


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
    if experiment_number in [0,2,3,4,5,6]:  # HCP experiments
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
    common_params['raw_save_root'] = '/media/mike/Files/Data and Results/N-back' if local_test else "/scratch/InfoDynFuncStruct/Mike/N-back/"
    common_params['compress'] = False  # Decides whether npz or npy file type is used to save the raw local TE values
    common_params['compute_p'] = compute_p
    common_params['save_every'] = None

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
    
    elif experiment_number == 4:  # HCP -- time-lagged MI (gaussian)
        run(i, save_folder = get_save_folder('HCP_MI-gaussian'), apply_global_mean_removal = True,
            trim_start = 50, trim_end = 25, set_k_to_0 = True, calc_type = 'gaussian', **common_params)

    elif experiment_number == 5:  # HCP -- time-lagged MI (KSG)
        run(i, save_folder = get_save_folder('HCP_MI-KSG'), apply_global_mean_removal = True,
            trim_start = 50, trim_end = 25, set_k_to_0 = True, **common_params)

    elif experiment_number == 6:  # HCP - using population parameters
        run(i, save_folder = get_save_folder('HCP_pop-param'), apply_global_mean_removal = True,
            trim_start = 50, trim_end = 25, **common_params)

    else:
        raise Exception("Experiment not defined")


############################################################################################################
if __name__ == "__main__":
    if os.path.exists('/home/mili7522/'):
        local_test = False
    else:
        local_test = True

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Run TE calculation on a particular data set.'
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
        print("Testing for one pair")
        test_for_one_pair()
