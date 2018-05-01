#Logistic model on Ad Purchase Conversions
#This will be based on the same test as Python Script except the model will be estimated by tensorflow instead of scikit-learn

import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf

#read data

reader = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/train_sample.csv', header=0, engine='c')


