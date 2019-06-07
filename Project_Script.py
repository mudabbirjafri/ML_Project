import matplotlib.pyplot as plt
%matplotlib inline
import random
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import datasets, svm, tree, preprocessing, metrics
import sklearn.ensemble as ske
import os
import datetime

os.chdir('C:/Users/adhillon/OneDrive - NVIDIA Corporation/AnacondaProjects/OPS 808/Dataset/')
nyc_data_df = pd.read_csv('nyc-rolling-sales.csv')
os.chdir('C:/Users/adhillon/Downloads')
geo_location = pd.read_csv('us-zip-code-latitude-and-longitude (2).csv', sep = ';')
geo_location['Zip'] = geo_location['Zip'].astype('int64')
geo_location_zip =  geo_location.set_index('Zip')


nyc_data = nyc_data_df

# Early preprocessing needed for graphing
nyc_data = nyc_data.replace(to_replace=' -  ', value ='0') # quick fix for dash
nyc_data['SALE PRICE'] = nyc_data['SALE PRICE'].astype('int64')
nyc_data['LAND SQUARE FEET'] = nyc_data['LAND SQUARE FEET'].astype('int64')
nyc_data['GROSS SQUARE FEET'] =nyc_data['GROSS SQUARE FEET'].astype('int64')

# examine the dataset
nyc_data.dtypes
nyc_data.shape
nyc_data.describe()

# map employee
geo_location_map_lat= geo_location_zip['Latitude']
geo_location_map_lat= geo_location_map_lat.to_dict()
geo_location_map_long = geo_location_zip['Longitude']
geo_location_map_long = geo_location_map_long.to_dict()
nyc_data['Longitude'] = nyc_data['ZIP CODE'].map(geo_location_map_long)
nyc_data['Latitude'] = nyc_data['ZIP CODE'].map(geo_location_map_lat)

# for visualization
nyc_data['BOROUGH'][nyc_data['BOROUGH']==1]='Manhattan'
nyc_data['BOROUGH'][nyc_data['BOROUGH']==2]='Bronx'
nyc_data['BOROUGH'][nyc_data['BOROUGH']==3]='Brooklyn'
nyc_data['BOROUGH'][nyc_data['BOROUGH']==4]='Queens'
nyc_data['BOROUGH'][nyc_data['BOROUGH']==5]='Staten Island'

# create bins
bins = [np.quantile(nyc_data['SALE PRICE'],0.2),np.quantile(nyc_data['SALE PRICE'],0.4), np.quantile(nyc_data['SALE PRICE'],0.5), np.quantile(nyc_data['SALE PRICE'],0.6), np.quantile(nyc_data['SALE PRICE'],0.80), np.quantile(nyc_data['SALE PRICE'],1)]
nyc_data['SALE_PRICE_BIN'] = pd.cut(nyc_data['SALE PRICE'], bins =bins,include_lowest=False)

# visualization
f, (ax1, ax2)= plt.subplots(2, figsize = [12,12])
fig1 = sns.scatterplot(x = 'Longitude', y = 'Latitude', hue = 'BOROUGH',
                style = 'BOROUGH',data=nyc_data,ax=ax1)
fig1.set_title('GEO REAL ESTATE MAP BY DISTRICT')
fig1.legend(loc = 'upper left')
fig2 = sns.scatterplot(x = 'Longitude', y = 'Latitude', hue = 'SALE_PRICE_BIN',
                style = 'SALE_PRICE_BIN',data=nyc_data, ax= ax2)
fig2.set_title('GEO REAL ESTATE MAP BY PRICE')
fig2.legend(loc = 'upper left')
plt.show()
# We notice Manhattan has the highest concentration of expensive properties; however, we also see Brooklyn and Queens having fairly expensive properties.
# Brooklyn and Queens could be possible areas to invest money


# Filter?
nyc_data = nyc_data[nyc_data['SALE PRICE']>100]

sns.boxplot(x='BOROUGH', y='SALE PRICE', data= nyc_data[nyc_data['SALE PRICE']<np.quantile(nyc_data['SALE PRICE'],0.9)], palette = 'Set3')




# create population tables
my_tables = {}
for field in nyc_data.columns:
    my_tables[field] = nyc_data.groupby([field]).count().iloc[:,:1]/len(nyc_data)


# preprocessing
nyc_data['SALE DATE'] = pd.to_datetime(nyc_data["SALE DATE"])
nyc_data.drop(['EASE-MENT', 'APARTMENT NUMBER'], axis=1, inplace=True) #Remove Empty columns
nyc_data =nyc_data.loc[:, ~nyc_data.columns.str.contains('^Unnamed')]


# So correlation
corr_list = nyc_data.corr()['SALE PRICE'].sort_values(ascending = False)
corr_list

# this does OneHotEncoder and LabelEncoder in one step
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
nyc_data['BUILDING CLASS CATEGORY'] = encoder.fit_transform(nyc_data['BUILDING CLASS CATEGORY'])
nyc_data['BOROUGH'] = encoder.fit_transform(nyc_data['BOROUGH'])

#Use LabelEncoder to convert labels into binary codes
from sklearn.preprocessing import LabelEncoder, StandardScaler
LE = LabelEncoder()
nyc_data['NEIGHBORHOOD'] = LE.fit_transform(nyc_data['NEIGHBORHOOD'])
nyc_data['BUILDING CLASS AT PRESENT'] = LE.fit_transform(nyc_data['BUILDING CLASS AT PRESENT'])
nyc_data['TAX CLASS AT TIME OF SALE'] = LE.fit_transform(nyc_data['TAX CLASS AT TIME OF SALE'])
nyc_data['TAX CLASS AT PRESENT'] = LE.fit_transform(nyc_data['TAX CLASS AT PRESENT'])
nyc_data['BUILDING CLASS AT TIME OF SALE'] = LE.fit_transform(nyc_data['BUILDING CLASS AT TIME OF SALE'])
nyc_data['BOROUGH'] = LE.fit_transform(nyc_data['BOROUGH'])
nyc_data['BUILDING CLASS CATEGORY'] = LE.fit_transform(nyc_data['BUILDING CLASS CATEGORY'])
nyc_data['YEAR BUILT'] = LE.fit_transform(nyc_data['YEAR BUILT']) # Approved
nyc_data['BLOCK'] = LE.fit_transform(nyc_data['BLOCK'])
nyc_data['NEIGHBORHOOD'] = LE.fit_transform(nyc_data['NEIGHBORHOOD'])

# Normalize Dataset
scaler = StandardScaler()
nyc_data['SALE PRICE'] = scaler.fit_transform(nyc_data['SALE PRICE'].values.reshape(-1,1))

# Create training and test data
from sklearn.model_selection import train_test_split
X = nyc_data[['GROSS SQUARE FEET','TAX CLASS AT PRESENT','BUILDING CLASS CATEGORY', 'BLOCK',
              'BOROUGH','TOTAL UNITS','BUILDING CLASS AT TIME OF SALE','TAX CLASS AT TIME OF SALE','YEAR BUILT']]
Y = nyc_data['SALE PRICE']
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=66, test_size=0.2)

# First model will begin with a multi linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
y_pred_test_inverse = scaler.inverse_transform(y_pred)
y_test_test_inverse = scaler.inverse_transform(y_test)


# Lets utilize a ensemble model
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test_test_inverse, y_pred_test_inverse)
mean_absolute_error(y_test_test_inverse, y_pred_test_inverse)

# Create keras model
from keras import models
from keras import layers
from keras import metrics
from keras import callbacks
from keras.layers import Dense
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(250, activation='relu',input_dim=len(X_train.columns)))
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(75, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')

history = model.fit(X_train, y_train,
                    epochs=5, batch_size=50,
                    validation_data = (X_test, y_test))
