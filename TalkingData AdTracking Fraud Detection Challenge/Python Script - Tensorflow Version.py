#Logistic model on Ad Purchase Conversions
#This will be based on the same test as Python Script except the model will be estimated by tensorflow instead of scikit-learn

import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf

#Load training data

data = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/train_sample.csv', header=0, engine='c')

#Clean Training data

#Creating Year, Month and Day Columns
data['Year2'] = data['click_time'].str[:4]
data['Month2'] = data['click_time'].str[5:7]
data['Day2'] = data['click_time'].str[8:10]
data['hour2'] = data['click_time'].str[11:13]
data['minute2'] = data['click_time'].str[14:16]
data['second2'] = data['click_time'].str[17:19]

#Delete click time and attributed time columns 
del data['click_time']
del data['attributed_time']

#Turn year, month and day columns from str to int
data['Year']=data['Year2'].astype(int)*1
data['Month']=data['Month2'].astype(int)*1
data['Day']=data['Day2'].astype(int)*1
data['Hour']=data['hour2'].astype(int)*1
data['Minute']=data['minute2'].astype(int)*1
data['Second']=data['second2'].astype(int)*1
del data['Year2']
del data['Month2']
del data['hour2']
del data['Day2']
del data['minute2']
del data['second2']

#Split train set and validation (test) set

train, test = train_test_split(data, test_size=0.3)

