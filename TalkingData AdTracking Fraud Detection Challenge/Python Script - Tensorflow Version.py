#Logistic model on Ad Purchase Conversions
#This will be based on the same test as Python Script except the model will be estimated by tensorflow instead of scikit-learn

import csv
import pandas
from sklearn.cross_validation import train_test_split
import tensorflow as tf

#create empty pandas dataframe to read CSV
headings = ['ip', 'app','device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
data = pd.DataFrame(columns = headings)

#read training data csv, append each row

with open('C:/Users/Tony Cai/Documents/Ad Prediction/train_sample.csv', 'r') as csvfile:
    dataset = csv.reader(csvfile)
    for row in dataset:
        if row[0] == "ip":
            continue
        else:
            data = data.append(pd.Series(row, index=headings), ignore_index=True)

