import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats as sm_stats

import datetime
import re
import category_encoders as ce
from sklearn.impute import SimpleImputer, KNNImputer
from numpy import percentile

import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, SelectFromModel
from sklearn.feature_selection import mutual_info_regression, chi2, f_regression, f_classif
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def phase1():
    labor = pd.read_csv("046/labor.csv", sep='\t')
    labor.rename(columns = {"Unnamed: 0": "index"}, inplace = True)
    labor = labor.drop(["index", "name"], axis=1)
    smoker_encoding = {"Y": 1, "N": 0, "yes": 1, "no": 0}
    labor["smoker"].replace(smoker_encoding, inplace=True)

    profiles = pd.read_csv("046/profiles.csv", sep='\t')
    profiles.rename(columns = {"Unnamed: 0": "index"}, inplace = True)
    profiles = profiles.drop(["index"], axis=1)
    profiles["race"].replace({"black": "Black", "white": "White", "blsck": "Black"}, inplace=True)
    profiles["birthdate"] = pd.to_datetime(profiles['birthdate'], utc=False)

    merged = pd.merge(profiles, labor, how='outer', on='ssn')
    merged = merged.drop(["ssn"], axis=1)
    return merged

class handleNA(TransformerMixin):
    def __init__(self, method, strategy=None):
        self.method = method
        self.strategy = strategy
        
    def removeNA(self, merged):
        return merged.dropna()

    def getNAcols(self, merged):
        return merged.columns[merged.isnull().any()].tolist()

    def replaceNaN(self, original_merged):
        na_cols = self.getNAcols(original_merged)
        strategy = self.strategy
        new_merged = original_merged.copy()
        if strategy == "kNN":
            imp_strategy = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    #         imp_strategy = KNNImputer()
        elif strategy == "mean" or strategy == "median":    
            imp_strategy = SimpleImputer(missing_values=np.nan, strategy=strategy)
        else:
            raise Exception("Unsupported strategy")
        for col in na_cols:
            new_merged[col] = imp_strategy.fit_transform(new_merged[[col]])
        return new_merged
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        if self.method == 'nothing':
            return X
        elif self.method == 'remove':
            return self.removeNA(X);
        elif self.method == 'replace':
            return self.replaceNaN(X)
        else:
            raise Exception("Unsupported method")
            
class handleOutliers(TransformerMixin):
    def __init__(self, method):
        self.method = method
        
    def onlyNumCols(self, merged):
        return merged.drop(["residence", "job", "company", "name", "birthdate"], axis=1, errors='ignore')

    def identify_outliers(self, merged):
        suma = 0;
        for col in merged.columns:
            q25, q75 = percentile(merged[col], 25), percentile(merged[col], 75)
            iqr = q75 - q25
            cut_off = iqr * 1.5
            lower, upper = q25 - cut_off, q75 + cut_off
            outliers = merged[((merged[col] < lower) | (merged[col] > upper))] 
            print(col, 'Identified outliers: %d' % len(outliers))
            suma += len(outliers)
        print('Sum of identified outliers: %d' % suma)

    def remove_outliers(self, merged):
        newMerged = merged.copy()
        for col in newMerged.columns:
            q25, q75 = percentile(newMerged[col], 25), percentile(newMerged[col], 75)
            iqr = q75 - q25
            cut_off = iqr * 1.5
            lower, upper = q25 - cut_off, q75 + cut_off
            newMerged = newMerged[((newMerged[col] >= lower) & (newMerged[col] <= upper))] 
        return newMerged

    def replace_outliers(self, merged):
        newMerged = merged.copy()
        for col in newMerged.columns:
            q05, q95 = percentile(newMerged[col], 5), percentile(newMerged[col], 95)
            newMerged[col] = np.where(newMerged[col] < q05, q05, newMerged[col])
            newMerged[col] = np.where(newMerged[col] > q95, q95, newMerged[col])
        return newMerged
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        if self.method == 'nothing':
            return self.onlyNumCols(X)
        elif self.method == 'remove':
            return self.remove_outliers(self.onlyNumCols(X))
        elif self.method == 'replace':
            return self.replace_outliers(self.onlyNumCols(X))
        else:
            raise Exception("Unsupported method")
            
class handleCategorical(TransformerMixin):
    def transformResidenceNLP(self, merged):
        for i in merged['residence'].index:
            country_code = re.findall('[A-Z]{2} [0-9]{5}', str(merged['residence'][i]))[0]
            merged.at[i, 'state']=re.findall('[A-Z]{2}', country_code)[0]
        len(merged['state'].value_counts())
        return merged.drop('residence', axis=1)

    def encodeOrdinal(self, merged):
        transformed = self.transformResidenceNLP(merged)
        ce_ordinal = ce.OrdinalEncoder(cols=['race', 'state', 'blood_group', 'relationship'])
        encoded = ce_ordinal.fit_transform(transformed)
        return encoded

    def frombirthtoage(self, born):
        now = datetime.date.today()
        return now.year - born.year - ((now.month, now.day) < (born.month, born.day))

    def computeAge(self, merged):
        ages = merged['birthdate'].apply(lambda d: self.frombirthtoage(d))
        merged = merged.assign(age=ages.values)
        return merged.drop('birthdate', axis=1)

    def encodeOneHot(self, merged):
        ce_OHE = ce.OneHotEncoder(cols=['sex'], use_cat_names=True)
        merged = ce_OHE.fit_transform(merged)
        return merged

    def fit(self, X):
        return self
    
    def transform(self, X):
        new_data = self.encodeOrdinal(X)
        new_data = self.computeAge(new_data)
        new_data = self.encodeOneHot(new_data)    
        return new_data
    
class handleTransformations(TransformerMixin):
    def __init__(self, method):
        self.method = method
        
    def transformPower(self, merged):
        power = PowerTransformer(method='yeo-johnson', standardize=True)
        merged_without_indicator = merged.drop('indicator', axis=1)
        indicator = merged['indicator']
        df_return = pd.DataFrame(power.fit_transform(merged_without_indicator), columns = merged_without_indicator.columns)
        df_return = pd.concat([df_return, indicator], axis=1)
        return df_return
    
    def transormQuan(self, merged):
        quan = QuantileTransformer(n_quantiles=10, random_state=0)
        merged_without_indicator = merged.drop('indicator', axis=1)
        indicator = merged['indicator']
        df_return = pd.DataFrame(quan.fit_transform(merged_without_indicator), columns = merged_without_indicator.columns)
        df_return = pd.concat([df_return, indicator], axis=1)
        return df_return
    
    def scaleMM(self, merged):
        norm_s = MinMaxScaler()
        merged_without_indicator = merged.drop('indicator', axis=1)
        indicator = merged['indicator']
        df_return = pd.DataFrame(norm_s.fit_transform(merged_without_indicator), columns = merged_without_indicator.columns)
        df_return = pd.concat([df_return, indicator], axis=1)
        return df_return
        
    def scaleS(self, merged):
        stan_s = StandardScaler()
        merged_without_indicator = merged.drop('indicator', axis=1)
        indicator = merged['indicator']
        df_return = pd.DataFrame(stan_s.fit_transform(merged_without_indicator), columns = merged_without_indicator.columns)
        df_return = pd.concat([df_return, indicator], axis=1)
        return df_return
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        if self.method == 'nothing':
            return X
        elif self.method == 'power':
            return self.transformPower(X)
        elif self.method == 'quan':
            return self.transormQuan(X)
        elif self.method == 'minmax':
            return self.scaleMM(X)
        elif self.method == 'standard':
            return self.scaleS(X)
        else:
            raise Exception("Unsupported method")

class handleSelection(TransformerMixin):
    def __init__(self, list_attributes):
        self.list_attributes = list_attributes
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        if self.list_attributes == 'all':
            return X
        else:
            return X[self.list_attributes]

def pipetofile(transformed_data, X_train, X_test, y_train, y_test):
    X_train.to_csv('processed_data_after_phase_2_X_train.csv', sep='\t')
    X_test.to_csv('processed_data_after_phase_2_X_test.csv', sep='\t')
    y_train.to_csv('processed_data_after_phase_2_y_train.csv', sep='\t')
    y_test.to_csv('processed_data_after_phase_2_y_test.csv', sep='\t')            