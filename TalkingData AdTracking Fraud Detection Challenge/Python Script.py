#Logistic model on Ad Purchase Conversions
#This is to create a logistics learning model on whether a user will download the advertised app after clicking into the ad
#In this version, we will estimate the model using Scikit-Learn

import time
start = time.time()

##############

import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

#create empty pandas dataframe to read CSV
headings = ['ip', 'app','device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
data = pd.DataFrame(columns = headings)

#read training data csv, append each row to pandas dataframe

with open('C:/Users/Tony Cai/Documents/Ad Prediction/train.csv', 'r') as csvfile:
    dataset = csv.reader(csvfile)
    for row in dataset:
        if row[0] == "ip":
            continue
        else:
            data = data.append(pd.Series(row, index=headings), ignore_index=True)

end = time.time()
        
#data = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/train.csv',header=0)

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

#Estimate model parameters by repeatedly randomly sampling 80% of training set to estimate model and average out resulting parameters 
for i in range(1, 5):
    
    #split training set by 70% train and 30% validation
    train, test = train_test_split(data, test_size=0.3)
    X_train = train[X_cols]
    Y_train = train['is_attributed']
    
    #train logistic model for said instance
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    
    #collecting/storing model parameters
    coefficients = coefficients.append(pd.DataFrame(logreg.coef_, columns = X_cols), ignore_index=True)

#Estimating parameters for the model by averaging the coefficients from repeated testing
model_param = coefficients.mean(axis=0)

print(model_param)

##########

end = time.time()
print(end-start)


