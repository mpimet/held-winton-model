## Held-Winton two layer model for EEI workshop

Historical ERF comes from Forster et al. (2025; https://essd.copernicus.org/articles/17/2641/2025/) covering 1850-2025 using IPCC methods (data from https://github.com/ClimateIndicator/forcing-timeseries)


Example code to read in dF, dT and dN

start_yr=2001
end_yr=2024

# Load in HadCRUT5 data
version='HadCRUT.5.0.2.0'
data=np.loadtxt(dir+''+version+'_Globalannual_1850-2024.txt',delimiter=',')
dT=data[1]
yr_dT=data[0]
i=np.where(yr_dT == start_yr)
j=np.where(yr_dT == end_yr)
dT=dT[i[0][0]:j[0][0]+1]

# Load in ERF data
df=pd.read_csv(dir+'ERF_best_DAMIP_1750-2024.csv')
dF=df['total'].to_numpy()
yr_dF=df['Unnamed: 0'].to_numpy()
k=np.where(yr_dF == 1850)
dF=dF-dF[k[0][0]]
i=np.where(yr_dF == start_yr)
j=np.where(yr_dF == end_yr)
dF=dF[i[0][0]:j[0][0]+1]
yr_dF=yr_dF[i[0][0]:j[0][0]+1]

# Load in CERES data
version='CERES-EBAF-TOA-Ed4.2.1'
data=np.loadtxt(dir+''+version+'_Globalannual_2001-2024.txt',delimiter=',')
dN=data[1]
dLW=data[2]
dSW=data[3]
yr=np.arange(len(dN))+2001
i=np.where(yr == start_yr)
j=np.where(yr == end_yr)
dN=dN[i[0][0]:j[0][0]+1]
dLW=dLW[i[0][0]:j[0][0]+1]
dSW=dSW[i[0][0]:j[0][0]+1]
yr=yr[i[0][0]:j[0][0]+1]
