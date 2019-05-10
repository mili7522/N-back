import jpype
import utils
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

history_lengths = range(1, 6)
delays = range(1, 2)

calc = utils.startCalc()

def plotAISAcrossParams(ais_values):
    plt.imshow(ais_values, extent = (0.5, len(delays)+0.5, len(history_lengths)+0.5, 0.5))
    plt.yticks(history_lengths); plt.xticks(delays); plt.xlabel('Delay'); plt.ylabel('History Length')
    plt.colorbar(label='AIS'); plt.tight_layout()
    plt.savefig('AIS_parameters.png', dpi = 200, bbox_inches='tight')
    plt.show()

def computeAIS(k, tau, data, calc = None, compute_local = False, compute_p = False):
    if calc is None:
        calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
        calc = calcClass()
    
    # calc.setProperty("k_HISTORY", str(k))
    # calc.setProperty("tau", str(tau))
    calc.initialise(k, tau)
    calc.setObservations(data)
    if compute_p:
        measDist = calc.computeSignificance(1000)
        # print("Surrogate Mean:", measDist.getMeanOfDistribution())
        # print("Surrogate Std:", measDist.getStdOfDistribution())
        p = measDist.pValue
    else:
        p = None
    if compute_local:
        return np.array( calc.computeLocalOfPreviousObservations() ), p
    else:
        return calc.computeAverageLocalOfObservations(), p  # KSG Estimator is already bias adjusted

def getLocalsForRegion(region_idx, df, print_max_idx = True, compute_p = False):
    data = df.iloc[region_idx].values
    data = utils.preprocess(data)
    
    ais_values = np.zeros((len(history_lengths), len(delays)))

    for i, history_length in enumerate(history_lengths):
        for j, delay in enumerate(delays):                   
            ais_values[i, j], _ = computeAIS(history_length, delay, data, calc, compute_local=False)

    max_idx = utils.getMaxIdx2D(ais_values, print_ = print_max_idx)

    result, p = computeAIS(history_lengths[max_idx[0]], delays[max_idx[1]], data, calc, compute_local=True, compute_p = compute_p)
    return result, ais_values, max_idx, p

def getLocalsForAllRegions(df, print_max_idx = True, parameters = None, compute_p = False):
    regions, timepoints = (333, 405)
    results = np.zeros((regions, timepoints))
    all_max_idx = np.zeros((regions, 2))
    p_values = np.zeros(regions)
    for region in range(regions):
        if parameters is None:
            results[region], _, max_idx, p_values[region] = getLocalsForRegion(region, df, print_max_idx, compute_p = compute_p)
            all_max_idx[region] = np.array(max_idx)
        else:
            p = parameters[region]
            data = df.iloc[region].values
            data = utils.preprocess(data)
            results[region], p_values[region] = computeAIS(history_lengths[p[0]], delays[p[1]], data, calc, compute_local=True, compute_p = compute_p)
    return results, all_max_idx, p_values

def getAverageParametersAcrossPatients():
    files = utils.getAllFiles()
    regions = 333
    all_max_idx = np.zeros((regions, 2))
    for region in range(regions):
        ais_values = np.zeros((len(files), len(history_lengths), len(delays)))
        for k, file in enumerate(files):
            df = utils.loadData(file)
            data = df.iloc[region].values
            data = utils.preprocess(data)
            for i, history_length in enumerate(history_lengths):
                for j, delay in enumerate(delays):           
                    ais_values[k, i, j], _ = computeAIS(history_length, delay, data, calc, compute_local=False)
        mean_ais_values = ais_values.mean(axis = 0)
        max_idx = utils.getMaxIdx2D(mean_ais_values)
        all_max_idx[region] = np.array(max_idx)
    return all_max_idx



if __name__ == "__main__":
    # filename = 100307
    # region_idx = 0
    # df = utils.loadData(filename = filename)
    # result, ais_values, max_idx, p_value = getLocalsForRegion(region_idx = region_idx, df = df, compute_p = True)
    # plotAISAcrossParams(ais_values)
    # plt.plot(result); plt.xlabel('Time'); plt.ylabel('AIS')
    # plt.title("AIS Kraskov: {}[{}]\nHistory = {}, Delay = {}".format(filename, region_idx, history_lengths[max_idx[0]], delays[max_idx[1]]))


    # for i, file in enumerate(utils.getAllFiles()):
    i = int(sys.argv[1])
    files = utils.getAllFiles()
    file = files[i]
    os.makedirs("Results/AIS Local - Individual Parameters/idx", exist_ok = True)
    os.makedirs("Results/AIS Local - Individual Parameters/p_values", exist_ok = True)
    os.makedirs("Results/AIS Local - Population Parameters/p_values", exist_ok = True)

    # if os.path.exists('Results/AIS Local - Individual Parameters/{}_AIS.csv'.format(utils.basename(file))):
    #     continue
    print("Processing", i, ":", file)
    # if os.path.exists('Results/AIS Local - Individual Parameters/p_values/{}.csv'.format(utils.basename(file))): exit()
    df = utils.loadData(file)
    results, all_max_idx, p_values = getLocalsForAllRegions(df, print_max_idx = False, compute_p = True)
    # pd.DataFrame(results).to_csv('Results/AIS Local - Individual Parameters/{}_AIS.csv'.format(utils.basename(file)), index = None, header = None)
    # pd.DataFrame(all_max_idx.astype(int)).to_csv('Results/AIS Local - Individual Parameters/idx/{}.csv'.format(utils.basename(file)), index = None, header = None)
    # pd.DataFrame(p_values).to_csv('Results/AIS Local - Individual Parameters/p_values/{}.csv'.format(utils.basename(file)), index = None, header = None)
    # utils.plotHeatmap(results, divergent = True)

#    if os.path.exists('Results/Population all_max_idx.csv'):
#        all_max_idx = pd.read_csv('Results/Population all_max_idx.csv', header = None).values
#    else:
#        all_max_idx = getAverageParametersAcrossPatients()
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


