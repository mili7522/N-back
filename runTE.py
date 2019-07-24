import glob
import os
import sys
import numpy as np
import pandas as pd
import utils


# calc = utils.startCalc(measure = 'te', estimator = 'ksg')
calc = utils.startCalc(measure = 'te', estimator = 'gaussian')

def computeTE(k, tau, acl, source_data, target_data, calc, source_history_length = 1,
              source_delay = 1, source_target_delay = 1,
              compute_local = False, compute_p = False, number_of_surrogates = 100):
    """
    Performs a calculation of Transfer Entropy using JIDT

    Arguments:
        k -- History length parameter
        tau -- Delay parameter
        acl -- Auto correlation length, used to set the dynamic correlation exclusion property
        source_data -- 1D Numpy array
        target_data -- 1D Numpy array
        calc -- The JIDT calculator
        compute_local -- If True, the local TE is returned instead of the average
        compute_p  -- If True, the p value of the average TE
        number_of_surrogates
    
    Returns:
        TE -- Either locals or average
        p -- p value. Returned if compute_p is true
    """
    calc.initialise(k, tau, source_history_length, source_delay, source_target_delay)
    calc.setProperty( 'BIAS_CORRECTION', 'true' )
    calc.setProperty("DYN_CORR_EXCL", str(acl))
    calc.setObservations(source_data, target_data)
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

def getLocalsForRegion(data, source_idx, target_idx, param_df, compute_p = False):
    """
    Calculates the local TE for a particular source region - target region pair, loading the parameters for target history length and target delay from file

    Arguments:
        data -- Numpy array. 2d array of shape (region, time). Preprocessing should have already been performed
        source_idx -- The index of the region under consideration, as an Int
        target_idx -- The index of the region under consideration, as an Int
        param_df -- 
        compute_p -- If True, computes the p value of the returned TE
    """
    source_data = data[source_idx]
    target_data = data[target_idx]
    
    history_length, delay = param_df.iloc[target_idx] + 1  # idx starts from 0, while parameters start from 1
    
    acl = max(utils.acl(source_data), utils.acl(target_data))  # Take the larger value from the source and target

    result, p = computeTE(history_length, delay, acl, source_data, target_data, calc, compute_local = True, compute_p = compute_p)
    return result, p

def getLocalsForAllRegions(data, param_df, compute_p = False, print_every = 50, save_every = 20, saver = None,
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
                results[source_idx, target_idx], p_values[source_idx, target_idx] = getLocalsForRegion(data, source_idx, target_idx, param_df, compute_p = compute_p)
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

if __name__ == "__main__":
    def test_for_one_pair():
        filename = '100307.tsv'
        path = '../Data'
        source_region = 1
        target_region = 0
        df, param_df = utils.loadData(filename, path, get_params = True, param_file = 'Results/HCP/AIS/idx/100307.csv')
        data = utils.preprocess(df, sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25)
        result, p_values = getLocalsForRegion(data, source_region, target_region, param_df, compute_p = False)
        utils.plotTimeseries(result)

    i = int(sys.argv[1])

    def run(i, data_path, extension, save_folder, raw_save_root = "/scratch/InfoDynFuncStruct/Mike/N-back/", GRP = False,
            compute_p = True, compress = False, set_k_to_0 = False, **preprocessing_params):
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
        results, p_values = getLocalsForAllRegions(data, param_df, compute_p = compute_p, saver = saver, results = results, p_values = p_values, idx_values = idx_values)

        # Add back the trimmed sections
        padding = ((0,0), (preprocessing_params.get('trim_start', 0), preprocessing_params.get('trim_end', 0)))
        saver.save_final_result(results, p_values, padding = padding, compress = compress)


    # HCP
    # run(i, data_path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP', #raw_save_root = '/media/mike/Files/Data and Results/N-back',
    #        sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25, compute_p = True, compress = False)

    # ATX
    # run(i, data_path = 'Data/ATX_data', extension = '.csv', save_folder = 'ATX', #raw_save_root = '/media/mike/Files/Data and Results/N-back',
    #        sampling_rate = 1, apply_global_mean_removal = True, trim_start = 25, trim_end = 25, compute_p = True, compress = False)

    # HCP -- no global mean removal
    # run(i, data_path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP_no-mean-removal',
    #        sampling_rate = 1.3, apply_global_mean_removal = False, trim_start = 50, trim_end = 25, compute_p = True, compress = False)
    
    # HCP -- using linear gaussian estimator
    run(i, data_path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP_gaussian', # raw_save_root = '/media/mike/Files/Data and Results/N-back',
           sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25, compute_p = True, compress = False)
    
    # HCP -- time-lagged MI (gaussian)
    run(i, data_path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP_MI_gaussian', # raw_save_root = '/media/mike/Files/Data and Results/N-back',
           sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25, set_k_to_0 = True, compute_p = True, compress = False)

    # HCP -- time-lagged MI (KSG)
    # run(i, data_path = 'Data/HCP', extension = '.tsv', save_folder = 'HCP_MI_KSG', # raw_save_root = '/media/mike/Files/Data and Results/N-back',
    #        sampling_rate = 1.3, apply_global_mean_removal = True, trim_start = 50, trim_end = 25, set_k_to_0 = True, compute_p = True, compress = False)
