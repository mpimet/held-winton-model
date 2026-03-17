import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load in ERF data
df=pd.read_csv('ERF_best_aggregates_1750-2024.csv')
dF=df['total'].to_numpy()
year=df['Unnamed: 0'].to_numpy()

# plot from 1850.5 onwards
k=np.where(year==1850.5)[0][0]
plt.plot(year[k:], dF[k:], label='Total ERF')
plt.show()
