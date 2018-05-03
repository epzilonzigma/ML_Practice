#Logistic model on Ad Purchase Conversions
#This will be based on the same test as Python Script except the model will be estimated by tensorflow instead of scikit-learn

import numpy as np
import pandas as pd
import tensorflow as tf

#create function which preps training data

def load_data(demo = True, train_size = 70000, test_size = 3000, chunk = 50000):
    
    #loading training data
    
    if demo == True: 
        data = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/train_sample.csv', header=0, engine='c')
    else:
        reader = pd.read_csv('C:/Users/Tony Cai/Documents/Ad Prediction/train_sample.csv', header=0, engine='c', chunksize = chunk)
        data = pd.DataFrame()
        for chunk in reader:
            data = data.append(chunk)
            
        
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
    
    data_cols = ['ip', 'app', 'device', 'os', 'channel', 'Day', 'Hour', 'Minute', 'Second', 'is_attributed']
    data = data[data_cols]
    
    #spliting datasets
    
    #create arrays of indices
    raw = np.arange(len(data))
    rando = np.copy(raw)
    
    #random rearrangement
    np.random.shuffle(rando)
    
    #create matrix of data and reshuffle the line items based on rando's rearrangement
    dataset = data.values
    dataset[raw, : ] = dataset[rando, :]
    
    #treate adn extract training set and validation set
    train_X = dataset[:train_size, :-1]
    train_Y = dataset[:train_size, -1]
    
    test_X = dataset[train_size:(train_size + test_size), :-1]
    test_Y = dataset[train_size:(train_size + test_size), -1]
    
    return train_X, train_Y, test_X, test_Y



#create script if this file is ran
    
if __name__ == '__main__':
    
    #load data
    Xtr, Ytr, Xte, Yte = load_data(demo = True)
    
    temp = Ytr.shape
    Ytr = Ytr.reshape(temp[0],1)
    temp = Yte.shape
    Yte = Yte.reshape(temp[0],1)
    
    #Variables Setup
    x = tf.placeholder(tf.float32, [None, 9])
    y = tf.placeholder(tf.float32, [None, 1])
    
    #Weights Setup
    w = tf.Variable(tf.zeros([9,1]))
    b = tf.Variable(tf.zeros([1]))
    
    #Setup prediction function/calculations
    def linearfn(X, w, b):
        z = tf.add(tf.matmul(X,w),b)
        return z
    
    zzz = linearfn(x, w, b)
    y_hat = tf.sigmoid(zzz)
    
    #setting up loss/cost function
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat)
    loss = tf.reduce_mean(entropy)
    
    #hyperparameter setup
    
    learning_rate = 0.0001
    total_epochs = 500
    
    #implement gradient descent with learning rate of 0.01
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(entropy)
    
    
    #initializer
    
    init = tf.global_variables_initializer()
    
    #running the session
    
    with tf.Session() as sess:
        
        #initialize all variables
        sess.run(init)
        
        #training model
        
        print('training...')
        
        for epochs in range(total_epochs):
            batch_x, batch_y = Xtr, Ytr
            feed = {x: batch_x, y: batch_y}
            sess.run([train_step,entropy],feed)
            
        var = tf.abs((y-y_hat))
        accuracy = tf.reduce_mean(tf.cast(var, tf.float32))
        print("Accuracy", accuracy.eval({x:Xte, y:Yte}))
        
        
        


                
            
    

