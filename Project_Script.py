import matplotlib.pyplot as plt
%matplotlib inline
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, svm, tree, preprocessing, metrics
import sklearn.ensemble as ske
import os
import datetime

os.chdir('C:/Users/adhillon/OneDrive - NVIDIA Corporation/AnacondaProjects/OPS 808/Dataset/')
nyc_data = pd.read_csv('nyc-rolling-sales.csv')
os.chdir('C:/Users/adhillon/Downloads')
geo_location = pd.read_csv('us-zip-code-latitude-and-longitude (2).csv', sep = ';')
geo_location['Zip'] = geo_location['Zip'].astype('int64')
geo_location_zip =  geo_location.set_index('Zip')

# Early preprocessing needed for graphing
nyc_data = nyc_data.replace(to_replace=' -  ', value ='0') # quick fix for dash
nyc_data['SALE PRICE'] = nyc_data['SALE PRICE'].astype('int64')
nyc_data['GROSS SQUARE FEET'] =nyc_data['GROSS SQUARE FEET'].astype('int64')
# map employee
geo_location_map_lat= geo_location_zip['Latitude']
geo_location_map_lat= geo_location_map_lat.to_dict()
geo_location_map_long = geo_location_zip['Longitude']
geo_location_map_long = geo_location_map_long.to_dict()
nyc_data['Longitude'] = nyc_data['ZIP CODE'].map(geo_location_map_long)
nyc_data['Latitude'] = nyc_data['ZIP CODE'].map(geo_location_map_lat)

nyc_data['SALE PRICE'].max()
# creating bins
nyc_data.describe()


bins = [np.quantile(nyc_data['SALE PRICE'],0.20),1, np.quantile(nyc_data['SALE PRICE'],0.4), np.quantile(nyc_data['SALE PRICE'],0.6), np.quantile(nyc_data['SALE PRICE'],0.80), np.quantile(nyc_data['SALE PRICE'],1)]
nyc_data['SALE_PRICE_BIN'] = pd.cut(nyc_data['SALE PRICE'], bins =bins,include_lowest=False)

f, (ax1, ax2) = plt.subplots(2, figsize = [16,12])
fig1 = sns.scatterplot(x = 'Longitude', y = 'Latitude', hue = 'BOROUGH',
                style = 'BOROUGH',data=nyc_data,ax=ax1).set_title('GEO REAL ESTATE MAP BY DISTRICT')
fig2 = sns.scatterplot(x = 'Longitude', y = 'Latitude', hue = 'SALE_PRICE_BIN',
                style = 'SALE_PRICE_BIN',data=nyc_data, ax= ax2)
fig2.set_title('GEO REAL ESTATE MAP BY PRICE')
fig2.legend(loc = 'upper left')
plt.show()

# examine the dataset
nyc_data.dtypes
nyc_data.shape

# create population tables
my_tables = {}
for field in nyc_data.columns:
    my_tables[field] = nyc_data.groupby([field]).count().iloc[:,:1]/len(nyc_data)

# preprocessing
nyc_data['SALE DATE'] = pd.to_datetime(nyc_data["SALE DATE"])
nyc_data['ZIP CODE'] = nyc_data['ZIP CODE'].astype('str')
nyc_data['BUILDING CLASS AT TIME OF SALE']=pd.Categorical(nyc_data['BUILDING CLASS AT TIME OF SALE']).codes
nyc_data['BUILDING CLASS AT TIME OF SALE']=nyc_data['BUILDING CLASS AT TIME OF SALE'].astype('category').cat.codes


nyc_data.drop(['EASE-MENT', 'APARTMENT NUMBER'], axis=1, inplace=True) #Remove Empty columns
nyc_data =nyc_data.loc[:, ~nyc_data.columns.str.contains('^Unnamed')]


#Use LabelEncoder to convert labels into binary codes
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
nyc_data['NEIGHBORHOOD'] = LE.fit_transform(nyc_data['NEIGHBORHOOD'])
nyc_data['BUILDING CLASS CATEGORY'] = LE.fit_transform(nyc_data['BUILDING CLASS CATEGORY'])
nyc_data['BUILDING CLASS AT PRESENT'] = LE.fit_transform(nyc_data['BUILDING CLASS AT PRESENT'])
nyc_data['TAX CLASS AT PRESENT'] = LE.fit_transform(nyc_data['TAX CLASS AT PRESENT'])
nyc_data['BUILDING CLASS AT TIME OF SALE'] = LE.fit_transform(nyc_data['BUILDING CLASS AT TIME OF SALE'])
nyc_data['ADDRESS'] = LE.fit_transform(nyc_data['ADDRESS'])

# create the model
from sklearn.model_selection import train_test_split
X = nyc_data.loc[:, nyc_data.columns != 'SALE PRICE']
Y = nyc_data['SALE PRICE']
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=66, test_size=0.2)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
x_scaled_train=scaler.transform(X_train)
x_scaled_test = scaler.transform(X_test)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])
