import glob
import os
import sys
import numpy as np
import pandas as pd
import utils
import argparse


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
              source_delay = 1, source_target_delay = 1,
              compute_local = False, compute_p = False, number_of_surrogates = 100, print_surrogate = True):
    """
    Performs a calculation of Transfer Entropy using JIDT

    Arguments:
        k -- History length parameter
        tau -- Delay parameter
        acl -- Auto correlation length, used to set the dynamic correlation exclusion property
        source_data -- 1D Numpy array containing the time series of the source region
        target_data -- 1D Numpy array containing the time series of the target region
        calc -- The JIDT calculator
        compute_local -- If True, the local TE is returned instead of the average
        compute_p  -- If True, the p value of the average TE is calculated and returned
        number_of_surrogates -- Number of surrogate distributions to test to estimate the p value
    
    Returns:
        TE -- Either locals or average
        p -- p value. Returned if compute_p is true, else None is returned
    """
    calc.initialise(k, tau, source_history_length, source_delay, source_target_delay)
    calc.setProperty( 'BIAS_CORRECTION', 'true' )
    calc.setProperty("DYN_CORR_EXCL", str(acl))
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
        return calc.computeAverageLocalOfObservations(), p  # KSG Estimator is already bias adjusted

def getLocalsForRegion(data, source_idx, target_idx, param_df, calc, compute_p = False):
    """
    Calculates the local TE for a particular source region - target region pair, loading the parameters for target history length and target delay from file

    Arguments:
        data -- Numpy array. 2d array of shape (region, time). Preprocessing should have already been performed
        source_idx -- The index of the region under consideration, as an Int
        target_idx -- The index of the region under consideration, as an Int
        param_df -- 
        calc -- The JIDT calculator
        compute_p -- If True, computes the p value of the returned TE
    """
    source_data = data[source_idx]
    target_data = data[target_idx]
    
    history_length, delay = param_df.iloc[target_idx] + 1  # idx starts from 0, while parameters start from 1
    
    acl = max(utils.acl(source_data), utils.acl(target_data))  # Take the larger value from the source and target

    result, p = computeTE(history_length, delay, acl, source_data, target_data, calc, compute_local = True, compute_p = compute_p)
    return result, p

def getLocalsForAllRegions(data, param_df, calc, compute_p = False, print_every = 50, save_every = 20, saver = None,
                           results = None, p_values = None, idx_values = None):
    """
    Calculates the local TE for all regions, by calling getLocalsForRegion

    Arguments:
        data -- Numpy array of shape (region, time). Preprocessing should have already been performed
        param_df
        compute_p
        print_every -- None, or Int giving the number of regions to calculate before printing an update of the progress
        save_every -- None, or Int giving the number of regions to calculate before saving the current state of the results
        saver -- TEResultSaver object, used to save the intermediate results
        results -- Loaded results from previous run
        p_values -- Loaded p_values from previous run
        idx_values -- Tuple of the last (source_idx, target_idx) of the saved results from previous run
    """
    regions, timepoints = data.shape
    if idx_values is None:
        starting_source_idx, starting_target_idx = 0, 0
        results = np.zeros((regions, regions, timepoints))
        p_values = np.zeros((regions, regions))
    else:
        starting_source_idx, starting_target_idx = idx_values
        assert results is not None and p_values is not None
    for source_idx in range(starting_source_idx, regions):
        for target_idx in range(regions):
            if source_idx == starting_source_idx and target_idx < starting_target_idx:
                continue
            if source_idx == target_idx:
                results[source_idx, target_idx] = np.nan
            else:
                if print_every is not None and (target_idx % print_every == 0) :
                    print(source_idx, "->", target_idx)
                results[source_idx, target_idx], p_values[source_idx, target_idx] = getLocalsForRegion(data, source_idx, target_idx, param_df, calc, compute_p = compute_p)
                if save_every is not None and saver is not None and (target_idx % save_every == 0):
                    saver.save_intermediate_result(results, p_values, (source_idx, target_idx + 1))
    return results, p_values

class TEResultSaver:
    def __init__(self, filename, save_folder, raw_save_root = "/scratch/InfoDynFuncStruct/Mike/N-back/"):
        """
        Arguments:
            filename -- Name of the files to be saved
            save_folder
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
            idx_values -- Tuple of (source_idx, target_idx) for the last processed source and target idx
        """
        self.save_raw(results, p_values, compress = False)
        with open(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(self.save_folder, self.filename)), 'w') as f:
            f.write(str(idx_values[0]) + "," + str(idx_values[1]))

    def save_final_result(self, results, p_values, padding = ((0,0), (0,0)), compress = False):
        """
        Saves the TE results - the raw array of shape (source_region, target_region, time) as well as the source and target TE to each region
        Arguments:
            results -- Numpy array containing the raw TE results
            p_values -- 
            padding -- Tuple of tuples containing the number of spaces to pad the TE results at the start and end of each dimension
            compress -- If True, save the raw results as a npz format instead of npy
        """
        self.save_raw(results, p_values, compress)

        target_te = np.nanmean(results, axis = 0)  # Average across all sources
        source_te = np.nanmean(results, axis = 1)  # Average across all targets

        # Add back the trimmed sections
        target_te = np.pad(target_te, padding, mode = 'constant', constant_values = 0)
        source_te = np.pad(source_te, padding, mode = 'constant', constant_values = 0)

        pd.DataFrame(target_te).to_csv('Results/{}/TE/In-Target/{}.csv'.format(self.save_folder, self.filename), index = None, header = None)
        pd.DataFrame(source_te).to_csv('Results/{}/TE/Out-Source/{}.csv'.format(self.save_folder, self.filename), index = None, header = None)
        
        # Clean up intermediate save files
        if compress:
            os.remove(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}.npy".format(self.save_folder, self.filename)))
            os.remove(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}.npy".format(self.save_folder, self.filename)))
        os.remove(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(self.save_folder, self.filename)))

    def save_raw(self, results, p_values, compress):
        if compress:
            np.savez_compressed(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}.npz".format(self.save_folder, self.filename)), results = results)
            np.savez_compressed(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}.npz".format(self.save_folder, self.filename)), p_values = p_values)
        else:
            # Save temp file first then rename, in case the process gets killed during the save
            np.save(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_temp.npy".format(self.save_folder, self.filename)), results)
            np.save(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_temp.npy".format(self.save_folder, self.filename)), p_values)
            os.replace(os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}_temp.npy".format(self.save_folder, self.filename)),
                       os.path.join(self.raw_save_root, "Results/{}/TE/raw/{}.npy".format(self.save_folder, self.filename)))
            os.replace(os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}_temp.npy".format(self.save_folder, self.filename)),
                       os.path.join(self.raw_save_root, "Results/{}/TE/raw/p_values/{}.npy".format(self.save_folder, self.filename)))


def test_for_one_pair(filename = '100307.tsv', path = '../Data', source_region = 1, target_region = 0,
                      param_file = 'Results/HCP/AIS/idx/100307.csv', calc_type = 'ksg'):
    calc = startCalc(calc_type)
    df, param_df = utils.loadData(filename, path, get_params = True, param_file = param_file)
    data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
    result, p_values = getLocalsForRegion(data, source_region, target_region, param_df, calc, compute_p = False)
    utils.plotTimeseries(result)

def run(i, data_path, extension, save_folder, raw_save_root = "/scratch/InfoDynFuncStruct/Mike/N-back/", GRP = False,
        compute_p = True, compress = False, set_k_to_0 = False, calc_type = 'ksg', **preprocessing_params):
    """
    Arguments:
        GRP -- True if processing the GRP data
        set_k_to_0 -- If True, skip loading of k and l parameters, instead initialising the DataFrame to -1 (so it gets set to 0 when one is added)
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
    
    print("Processing", i, ":", filename)
    if os.path.isfile(os.path.join(raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(save_folder, filename))):
        # Load previous results
        results = np.load(os.path.join(raw_save_root, "Results/{}/TE/raw/{}.npy".format(save_folder, filename)))
        p_values = np.load(os.path.join(raw_save_root, "Results/{}/TE/raw/p_values/{}.npy".format(save_folder, filename)))
        with open(os.path.join(raw_save_root, "Results/{}/TE/raw/{}_current_idx.txt".format(save_folder, filename)), 'r') as f:
            idx_values = f.readline()
        idx_values = map(int,idx_values.split(','))
    else:
        results, p_values, idx_values = None, None, None
        if glob.glob(os.path.join(raw_save_root, "Results/{}/TE/raw/p_values/{}.np*".format(save_folder, filename))):  # Check both compressed and uncompressed
            print("Result already present")
            exit()
    param_file = 'Results/{}/AIS/idx/{}.csv'.format(save_folder, filename)
    if set_k_to_0:
        df = utils.loadData(file, get_params = False, subject_id = subject_id)
        param_df = pd.DataFrame( np.zeros((len(df), 2), dtype = int) - 1 )
    else:
        df, param_df = utils.loadData(file, get_params = True, param_file = param_file, subject_id = subject_id)
    data = utils.preprocess(df, **preprocessing_params)
    saver = TEResultSaver(filename, save_folder, raw_save_root)
    calc = startCalc(calc_type)
    results, p_values = getLocalsForAllRegions(data, param_df, calc, compute_p, saver = saver, results = results, p_values = p_values, idx_values = idx_values)

    # Add back the trimmed sections
    padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
    saver.save_final_result(results, p_values, padding = padding, compress = compress)


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
    if experiment_number in [0,2,3,4,5]:  # HCP experiments
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
    common_params['compute_p'] = compute_p

    def get_save_folder(folder_name):
        save_folder  = folder_name + ('_with-p' if compute_p else "")
        save_folder += ('_r{:02}'.format(repetition) if repetition is not None else "")
        return save_folder

    # Run experiment
    print("Running experiment:", experiment_number)
    if experiment_number == 0:    # HCP
        run(i, save_folder = get_save_folder('HCP'), apply_global_mean_removal = True,
               trim_start = 50, trim_end = 25, compress = False, **common_params)

    elif experiment_number == 1:  # ATX
        run(i, save_folder = get_save_folder('ATX'), apply_global_mean_removal = True,
               trim_start = 25, trim_end = 25, compress = False, **common_params)

    elif experiment_number == 2:  # HCP -- no global mean removal
        run(i, save_folder = get_save_folder('HCP_no-mean-removal'), apply_global_mean_removal = False,
               trim_start = 50, trim_end = 25, compress = False, **common_params)
        
    elif experiment_number == 3:  # HCP - using linear gaussian estimator
        run(i, save_folder = get_save_folder('HCP_gaussian'), apply_global_mean_removal = True,
               trim_start = 50, trim_end = 25, compress = False, calc_type = 'gaussian', **common_params)
    
    elif experiment_number == 4:  # HCP -- time-lagged MI (gaussian)
        run(i, save_folder = get_save_folder('HCP_MI-gaussian'), apply_global_mean_removal = True,
               trim_start = 50, trim_end = 25, set_k_to_0 = True, compress = False, calc_type = 'gaussian', **common_params)

    elif experiment_number == 5:  # HCP -- time-lagged MI (KSG)
        run(i, save_folder = get_save_folder('HCP_MI-KSG'), apply_global_mean_removal = True,
               trim_start = 50, trim_end = 25, set_k_to_0 = True, compress = False, **common_params)
    
    else:
        raise Exception("Experiment not defined")


if __name__ == "__main__":
    if os.path.exists('/home/mili7522/'):
        local_test = False
    else:
        local_test = True

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Run TE calculation on a particular data set')
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
        print("Testing for one pair")
        test_for_one_pair()
