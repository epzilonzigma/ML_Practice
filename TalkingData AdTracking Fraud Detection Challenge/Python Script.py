#Logistic model on Ad Purchase Conversions
#This is to create a logistics learning model on whether a user will download the advertised app after clicking into the ad
#In this version, we will estimate the model using Scikit-Learn
#Date parsing coding contributed by Jason Ip (not owner) 

##############

import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

#load training dataset at 300,000 rows at a time

CHUNKSIZE = 300000
reader = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/train.csv', header=0, engine='c', chunksize=CHUNKSIZE)

data = pd.DataFrame()

for chunk in reader:
    data = data.append(chunk)

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

#Overview of size of dataframe
print(data.shape)
print(list(data.columns))

#Looks at headers of dataframe
data.head

X_cols = ['ip', 'app', 'device', 'os', 'channel', 'Day', 'Hour', 'Minute', 'Second']

#Model estiamtion

#Create an array to house coefficients
coefficients = pd.DataFrame(columns = X_cols)
intercepts = pd.DataFrame(columns = ['intercept'])

#Estimate model parameters by repeatedly randomly sampling 80% of training set to estimate model and average out resulting parameters 
simulation_iterations = 10

for i in range(1, simulation_iterations+1):
    
    #split training set by 70% train and 30% validation
    train, test = train_test_split(data, test_size=0.3)
    X_train = train[X_cols]
    Y_train = train['is_attributed']
    
    #train logistic model for said instance
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    
    #collecting/storing model parameters
    coefficients = coefficients.append(pd.DataFrame(logreg.coef_, columns = X_cols), ignore_index=True)
    intercepts = intercepts.append(pd.DataFrame(logreg.intercep_, columns = ['intercept']), ignore_index=True)

#Estimating parameters for the model by averaging the coefficients from repeated testing
model_param = coefficients.mean(axis=0)
intercept = float(intercepts.mean(axis=0))

param = tuple(model_param)

print(model_param)

#########

#Prediction of Test data

#Define Logistic Function for prediction

def logit(z):
    return 1/(1+np.exp(-1*z))

#load test data at 300,000 rows at a time

test_reader = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/test.csv', header=0, engine='c', chunksize=CHUNKSIZE)

test_data = pd.DataFrame()

for chunk in test_reader:
    test_data = test_data.append(chunk)

#Creating Year, Month and Day Columns
test_data['Year2'] = test_data['click_time'].str[:4]
test_data['Month2'] = test_data['click_time'].str[5:7]
test_data['Day2'] = test_data['click_time'].str[8:10]
test_data['hour2'] = test_data['click_time'].str[11:13]
test_data['minute2'] = test_data['click_time'].str[14:16]
test_data['second2'] = test_data['click_time'].str[17:19]

#Delete click time and attributed time columns 
del test_data['click_time']

#Turn year, month and day columns from str to int
test_data['Year']=test_data['Year2'].astype(int)*1
test_data['Month']=test_data['Month2'].astype(int)*1
test_data['Day']=test_data['Day2'].astype(int)*1
test_data['Hour']=test_data['hour2'].astype(int)*1
test_data['Minute']=test_data['minute2'].astype(int)*1
test_data['Second']=test_data['second2'].astype(int)*1
del test_data['Year2']
del test_data['Month2']
del test_data['hour2']
del test_data['Day2']
del test_data['minute2']
del test_data['second2']

#Create dataframe to store predicted values

pred = pd.DataFrame()

#Create an matrix for test inputs
test_input = np.asmatrix(test_data[X_cols].values)

#Calculation of predicted values
z = np.add(np.matmul(test_input, param),intercept)
predicted_values = logit(z)

#create submission file

#create results array
click_id = pd.DataFrame(test_data['click_id'], columns = ['click_id'])

predicted_values = predicted_values.transpose(1,0)
predicted_values = np.asarray(predicted_values)
predicted = np.asarray(tuple(predicted_values))
pred = pd.DataFrame(predicted, columns = ['is_attributed'])

results = pd.concat([click_id, pred], axis = 1)


#Publish results as CSV

with open('C:/Users/Tony Cai/Documents/Ad Prediction/submission.csv','w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows([['click_id','is_attributed']])
    writer.writerows(results.values)
    


