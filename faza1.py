import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats as sm_stats

filename_labor = "046/labor.csv"
labor = pd.read_csv(filename_labor, sep='\t')
filename_profiles = "046/profiles.csv"
profiles = pd.read_csv(filename_profiles, sep='\t')

labor["smoker"].replace({"Y": 1, "N": 0, "yes": 1, "no": 0}, inplace=True)
labor.rename(columns = {"Unnamed: 0": "index"}, inplace = True)

profiles["race"].replace({"black": "Black", "white": "White", "blsck": "Black"}, inplace=True)

profiles['birthdate'] = pd.to_datetime(profiles['birthdate'], utc=False)

labor2 = labor.drop(["index", "name"], axis=1)
profiles2 = profiles.drop(["job", "company", "name"], axis=1)

merged = pd.merge(profiles2, labor2, how='outer', on='ssn')