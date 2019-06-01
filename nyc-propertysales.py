import matplotlib.pyplot as plt
%matplotlib inline
import random
import numpy as np
import pandas as pd
import os
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from scipy import stats


os.chdir('C:/Users/Mudabbir/Desktop/Spring Quarter/Machine Learning 808/nyc-property-sales')
nyc_data = pd.read_csv('nyc-rolling-sales.csv')

# examine the dataset
nyc_data.dtypes
nyc_data.shape
nyc_data.isnull().sum()

nyc_data.groupby(['TAX CLASS AT PRESENT']).count().iloc[:,:1]/len(nyc_data)
nyc_data.groupby(['BOROUGH']).count().iloc[:,:1]/len(nyc_data)

nyc_data.head()

# preprocessing
nyc_data['SALE DATE'] = pd.to_datetime(nyc_data["SALE DATE"])
nyc_data = nyc_data.replace(to_replace=' -  ', value ='0')
nyc_data['SALE PRICE'] = pd.to_numeric(nyc_data['SALE PRICE'], errors='coerce')
nyc_data['BOROUGH'] = nyc_data['BOROUGH'].astype('category')
nyc_data['BUILDING CLASS AT TIME OF SALE']=pd.Categorical(nyc_data['BUILDING CLASS AT TIME OF SALE']).codes
nyc_data['BUILDING CLASS AT TIME OF SALE']=nyc_data['BUILDING CLASS AT TIME OF SALE'].astype('category').cat.codes
nyc_data['TAX CLASS AT TIME OF SALE'] = nyc_data['TAX CLASS AT TIME OF SALE'].astype('category')
nyc_data['TAX CLASS AT PRESENT'] = nyc_data['TAX CLASS AT PRESENT'].astype('category')
nyc_data['LAND SQUARE FEET'] = pd.to_numeric(nyc_data['LAND SQUARE FEET'], errors='coerce')
nyc_data['GROSS SQUARE FEET']= pd.to_numeric(nyc_data['GROSS SQUARE FEET'], errors='coerce')
nyc_data.drop(['EASE-MENT'], axis=1, inplace=True) #Remove Empty columns
nyc_data =nyc_data.loc[:, ~nyc_data.columns.str.contains('^Unnamed')]
sum(nyc_data.duplicated(nyc_data.columns))
nyc_data = nyc_data.drop_duplicates(nyc_data.columns, keep='last')
sum(nyc_data.duplicated(nyc_data.columns))

#checking missing values
nyc_data.columns[nyc_data.isnull().any()]
miss=nyc_data.isnull().sum()/len(nyc_data)
miss=miss[miss>0]
miss.sort_values(inplace=True)
print(miss)



#Use LabelEncoder to convert labels into binary codes
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
nyc_data['NEIGHBORHOOD'] = LE.fit_transform(nyc_data['NEIGHBORHOOD'])
nyc_data['BUILDING CLASS CATEGORY'] = LE.fit_transform(nyc_data['BUILDING CLASS CATEGORY'])
nyc_data['BUILDING CLASS AT PRESENT'] = LE.fit_transform(nyc_data['BUILDING CLASS AT PRESENT'])
nyc_data['TAX CLASS AT PRESENT'] = LE.fit_transform(nyc_data['TAX CLASS AT PRESENT'])
nyc_data['BUILDING CLASS AT TIME OF SALE'] = LE.fit_transform(nyc_data['BUILDING CLASS AT TIME OF SALE'])
nyc_data['ADDRESS'] = LE.fit_transform(nyc_data['ADDRESS'])
# nyc_data['SALE PRICE'] = nyc_data['SALE PRICE'].astype(str)

# nyc_data.fillna(nyc_data.mean(), inplace=True)
nyc_data.head()

Avg_price = nyc_data.groupby(['TAX CLASS AT PRESENT', 'YEAR BUILT'])['SALE PRICE'].mean()

Avg_price.head()

##Lets do a simple linear regression model
######
