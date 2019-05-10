import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### Individual params
all_max_idx = utils.loadArrays(path = '../Results/AIS Local - Individual Parameters/idx/').astype(int)

hist_len = pd.DataFrame(all_max_idx[:,:,0].ravel()) + 1
sns.countplot(data = hist_len, x = 0)
plt.xlabel('History Lengths'); plt.ylabel('Counts'); plt.show()

delay = pd.DataFrame(all_max_idx[:,:,1].ravel()) + 1
sns.countplot(data = delay, x = 0)
plt.xlabel('Delay'); plt.ylabel('Counts'); plt.show()


### Population params
all_max_idx = pd.read_csv('../Results/Population all_max_idx.csv', header = None).values

hist_len = pd.DataFrame(all_max_idx[:,0].ravel()) + 1
sns.countplot(data = hist_len, x = 0)
plt.xlabel('History Lengths'); plt.ylabel('Counts'); plt.show()

delay = pd.DataFrame(all_max_idx[:,1].ravel()) + 1
sns.countplot(data = delay, x = 0)
plt.xlabel('Delay'); plt.ylabel('Counts'); plt.show()
