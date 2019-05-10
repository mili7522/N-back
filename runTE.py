import jpype
import utils
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

source_history_length = 1
source_delay = 1
source_target_delay = 1

calc = utils.startCalc(measure = 'te')

def computeTE(k, tau, source_data, target_data, calc, compute_local = False, compute_p = False):
    calc.initialise(k, tau, source_history_length, source_delay, source_target_delay)
    calc.setObservations(source_data, target_data)
    if compute_p:
        measDist = calc.computeSignificance(100)
        # print("Surrogate Mean:", measDist.getMeanOfDistribution())
        # print("Surrogate Std:", measDist.getStdOfDistribution())
        p = measDist.pValue
    else:
        p = None
    if compute_local:
        return np.array( calc.computeLocalOfPreviousObservations() ), p
    else:
        return calc.computeAverageLocalOfObservations(), p  # KSG Estimator is already bias adjusted

def getLocalsForRegion(source_idx, target_idx, df, param_df, compute_p = False):
    source_data = df.iloc[source_idx].values
    source_data = utils.preprocess(source_data)

    target_data = df.iloc[target_idx].values
    target_data = utils.preprocess(target_data)
    
    history_length, delay = param_df.iloc[target_idx] + 1  # idx starts from 0, while parameters start from 1
    
    result, p = computeTE(history_length, delay, source_data, target_data, calc, compute_local = True, compute_p = compute_p)
    return result, p

def getLocalsForAllRegions(df, param_df, compute_p = False):
    regions, timepoints = (333, 405)
    results = np.zeros((regions, regions, timepoints))
    p_values = np.zeros((regions, regions))
    for source_idx in range(regions):
        for target_idx in range(regions):
            if source_idx == target_idx:
                results[source_idx, target_idx] = np.nan
            else:
                print(source_idx, target_idx)
                results[source_idx, target_idx], p_values[source_idx, target_idx] = getLocalsForRegion(source_idx, target_idx, df, param_df, compute_p = compute_p)
    return results, p_values

def saveTEResults(results, file, compress = False):
    # os.makedirs("Results/TE/raw", exist_ok = True)
    os.makedirs("/scratch/InfoDynFuncStruct/Mike/N-back/Results/TE/raw", exist_ok = True)
    os.makedirs("Results/TE/In-Target", exist_ok = True)
    os.makedirs("Results/TE/Out-Source", exist_ok = True)

    if compress:
        np.savez_compressed("/scratch/InfoDynFuncStruct/Mike/N-back/Results/TE/raw/{}.npz".format(utils.basename(file)), results = results)
    else:
        np.save("/scratch/InfoDynFuncStruct/Mike/N-back/Results/TE/raw/{}.npy".format(utils.basename(file)), results)
    
    target_te = np.nanmean(results, axis = 0)  # Average across all sources
    source_te = np.nanmean(results, axis = 1)  # Average across all targets

    pd.DataFrame(target_te).to_csv('Results/TE/In-Target/{}.csv'.format(utils.basename(file)), index = None, header = None)
    pd.DataFrame(source_te).to_csv('Results/TE/Out-Source/{}.csv'.format(utils.basename(file)), index = None, header = None)

if __name__ == "__main__":
    
    i = int(sys.argv[1])
    files = utils.getAllFiles()
    file = files[i]
    df, param_df = utils.loadData(file, get_params = True)
    # result, _ = getLocalsForRegion(source_idx = 1, target_idx = 0, df = df, param_df = param_df, compute_p = False)
    results, p_values = getLocalsForAllRegions(df, param_df, compute_p = False)
    saveTEResults(results, file, compress = False)
    
    
#    results, all_max_idx, p_values = getLocalsForAllRegions(df, print_max_idx = False, compute_p = False)