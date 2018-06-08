# -*- coding: utf-8 -*-
"""
This code is used to construct an ensemble classifier on the training set constructed from the data cleaning file

"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statistics import mean, median
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, average_precision_score


'''
Create a class for obtaining summary statistics from a series
'''

class Descriptive_Statistic:
    
    ###generates statistics for the targeted series
    def __init__(self, series, name):
        self.data = pd.Series(series, name = name)
        self.mean = mean(series)
        self.median = median(series)
        self.count = len(series)
        self.max = max(series)
        self.min = min(series)
        self.first_quartile = np.percentile(series, 25)
        self.third_quartile = np.percentile(series, 75) 
        self.name = name
    
    ###generate percentiles when needed
    
    def percentile(self, x):
        percentile = np.percentile(self.series, x)
        return percentile

###class for housing model diagnostics for classifiers
class model_diagnostics:
    
    #initialize class with predictions of input test sets
    def __init__(self, clf, X, y):
        self.clf = clf
        self.X = X
        self.y = y
        self.clf.fit(X,y)
        self.y_pred = self.clf.predict(X)
        
        #calculate diagnostic metrics
        self.accuracy = accuracy_score(self.y, self.y_pred)
        self.avg_precision = average_precision_score(self.y, self.y_pred)
        self.f1 = f1_score(self.y, self.y_pred)
        self.FPR, self.TPR, self.thresholds = roc_curve(self.y, self.y_pred)
        self.auc = auc(self.FPR, self.TPR)
        self.summary = [self.accuracy, self.avg_precision, self.auc, self.f1]
        

### build function to extract all important features of a classifier
        
def extract_factors(features, clf):
    
    a = []
    
    for i in range(0, len(features)):
        if best_AdaBoost_clf.feature_importances_[i] != 0:
            a.append([features[i], clf.feature_importances_[i]])

    return a


"""
Training and analyzing cleaned dataset
"""

if __name__ == "__main__":
    
    '''load cleaned data'''
    
    train_dir = "director of cleaned training data from 'data preparation' file"
    
    train_data = pd.read_csv(train_dir)
    train = train_data.dropna()
    
    
    '''
    generate descriptive statistics for loaded data
    
    1 - get summary statistics
    2 - look at this distribution of sales volume
    3 - look at members deemed performing POS vs non-performing POS
    
    '''
    
    volume = Descriptive_Statistic(train_data['volume'], 'Sales Volume')
    
    annualized_volume = Descriptive_Statistic(train_data['annualized_volume'], 'Annualized Sales Volume')
    
    train_volume = Descriptive_Statistic(train['volume'], 'Training Set Sales Volume')
    
    train_annualized_volume = Descriptive_Statistic(train['annualized_volume'], 'Training Set Annualized Sales Volume')
    
    POS_count = len(train_data)
    train_POS_count = len(train)
    
    SPPD = volume.mean
    annualized_SPPD = annualized_volume.mean
    train_SPPD = train_volume.mean
    train_annualized_SPPD = train_annualized_volume.mean
    
    High_performing_POS_count = train_data['performing_POS'].sum()
    Low_performing_POS_count = POS_count - High_performing_POS_count
    
    train_High_performing_POS_count = train['performing_POS'].sum()
    train_Low_performing_POS_count = train_POS_count - train_High_performing_POS_count
    
    
    ###Generate histogram to compare volume and annualized volume treatments and SPPD (standard for performance) locations in pre- and post- treatments###
    
    bins = range(0, 250000, 25000)
    
    #in full data
    plt.figure(figsize=(13, 8.5), edgecolor='white', frameon = 0)
    plt.hist([train_data['volume'],train_data['annualized_volume']], bins = bins, color = ['C0', 'C2'], label = ['Sales Volume','Annualized Sales Volume'])
    plt.plot([SPPD,SPPD],[0,700],'k-', color='C7', linestyle = ':')
    plt.plot([annualized_SPPD,annualized_SPPD],[0,700],'k-',color='k', lw=3)
    plt.legend(loc='upper right', fontsize=20)
    plt.xlabel('Volume', fontsize=24)
    plt.ylabel('Store Count', fontsize = 24)
    plt.title('Distribution of Sales Volume', fontsize=28)
    plt.tick_params(top='off', right='off', labelsize=18)
    plt.savefig('Sales_histogram.png')
    plt.show()
    
    #in training set used for modeling
    plt.figure(figsize=(13, 8.5), edgecolor='white', frameon = 0)
    plt.hist([train['volume'],train['annualized_volume']],color=['C0','C2'], bins = bins, label = ['Sales Volume','Annualized Sales Volume'])
    plt.plot([train_SPPD,train_SPPD],[0,500],'k-',color='C7', linestyle = ':')
    plt.plot([train_annualized_SPPD,train_annualized_SPPD],[0,500],'k-',color='k',lw=3)
    plt.legend(loc='upper right', fontsize=20)
    plt.xlabel('Volume', fontsize=24)
    plt.ylabel('Store Count', fontsize=24)
    plt.title('Distribution of Sales Volume with Amenities Data', fontsize=28)
    plt.tick_params(top='off', right='off', labelsize=18)
    plt.savefig('train_sales_histogram.png')
    plt.show()
    
    '''
    preparing datasets - input and output to be trained
    '''
    
    ###preparing output series and category labels
    
    y = train['performing_POS']
    
    categories = ['Low Performing POS', 'High Performing POS']
    
    ###indentify if there are any explanatory variables (features) to be dropped from input
    
    #drop columns/features with same value
    
    headers_to_drop = []
    
    for i in range(7, len(train.columns)-1):
        
        if train[train.columns[i]].sum() == 0:
            headers_to_drop.append(train.columns[i])
    

    XXX = train.drop(headers_to_drop, axis = 1)
    
    #Create input array X of surrounding amenities features
    
    X = XXX[XXX.columns[7:]]
    
    '''
    Exploratory data analysis using various tree/forest classification models
    '''
    
    #Instead of splitting training sets, we will be using 5-fold cross validation to select models where the train/test splits will occur during the process
    #Ideally, we would still like to split train/test sets and perform k-fold cross validation on train sets, but due to data constraints and class imbalance for target variable, this will be skipped for better heuristics gathering
    

    scoring = 'average_precision'
    folds = 5
    
    ###user cross-validation to identify which is better measure for decision trees exploration analysis (gini vs entropy)
    ###would like to double check to ensure that both algorithms yield near identifical prediction power to justify usage of gini over entropy due to computation efficiency
    
    clf_parameters = {'max_depth':[2,3,4,5,6,7,8,9,10,15,20,40], 'min_samples_leaf': [1,2,4], 'min_samples_split': [2,5,10], 'criterion':['gini','entropy'], 'random_state':[0]}
    
    clf_Rand = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions = clf_parameters, cv = folds, verbose = 2, scoring = scoring, n_iter=100)
    clf_Rand.fit(X,y)
    
    clf_Rand_params = clf_Rand.best_params_
    clf = clf_Rand.best_estimator_
    
    clf.fit(X,y)
    
    tree = export_graphviz(clf, out_file = 'DTC.dot', feature_names = X.columns, class_names = categories, filled = True)
    
    '''
    
    The following approach will be used for model selection:
        1) Random Search CV function will be used to ball park suitable model hyperparameters for each classifier
        2) Upon identifying the optimal model from Random Search Grid Search will be used for surrounding parameters to fine tune/optimize hyperparameters for the model
        3) Resulting classifier from Grid Search will be the classifier used for analysis.
    
    Note: due to time constraint and the differences of hyperparameters applicable for each classifier, the values entered for Grid Search are hard coded and reflecting of what
    was seen in the result of random search. The random search parameter is usually used as a median if not a minimum of the grid search set    
    
    Identification of key features will be examined via gini importance of classifier for each feature.
    
    '''
    
    '''
    Identifying key attributes via Random Forest classifier
    '''

    ###construct an Random Forest classifier using random search for ball parking good hyperparameters with respect to Precison Recall Curve as selection criteria
    
    RF_Rand_parameters = {'max_depth':[2,3,4,5,6,7,8,9,10,15,20,30,50], 'min_samples_leaf': [1,2,4], 'min_samples_split': [2,5,10],'n_estimators':[10,20,50,75,100,150,200], 'criterion':['gini','entropy'], 'random_state':[0]}
    
    RF_rand_clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions = RF_Rand_parameters, cv = folds, verbose = 2, scoring = scoring, n_iter=300)
    RF_rand_clf.fit(X,y)
    
    RF_rand_clf_params = RF_rand_clf.best_params_
    
    print(RF_rand_clf_params)
    
    ###construct an Random Forest classifier using grid search for hyperparameter tuning based in random grid search results
    
    RF_parameters = {'max_depth':[7,8,9,10,11], 'n_estimators':[40,45,50,55,60], 'min_samples_split': [2,3,4], 'min_samples_leaf': [3,4,5], 'criterion':['gini'], 'random_state':[0]}
    
    RF_clf = GridSearchCV(RandomForestClassifier(), param_grid = RF_parameters, scoring = scoring, cv=folds)
    RF_clf.fit(X,y)
    
    
    #identify the best Random Forest classifier in grid search
    best_RF_clf = RF_clf.best_estimator_
    
    #identify the most important factor in the best Random Forest classifier
    RF_features = extract_factors(X.columns, best_RF_clf)
    print(RF_features)
    
    #Calculate model diagnostics of Best Random Forest Classifier
    
    RF_diagnostics = model_diagnostics(best_RF_clf, X, y)
    
    #note read: Accuracy, Precision, AUC, f1 <-- this applies for all in summary prints going forward
    print(RF_diagnostics.summary)
    
    
    '''
    Search for the most important features using AdaBoost
    '''

    ###construct an AdaBoost classifier using random search for ball parking good hyperparameters with respect to Precison Recall Curve as selection criteria
    
    AdaBoost_Rand_parameters = {'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],'n_estimators':[10,20,50,75,100,150,200], 'algorithm':['SAMME','SAMME.R'], 'random_state':[0]}
    
    AdaBoost_rand_clf = RandomizedSearchCV(AdaBoostClassifier(), param_distributions = AdaBoost_Rand_parameters, cv = folds, verbose = 2, scoring = scoring, n_iter=100)
    AdaBoost_rand_clf.fit(X,y)
    
    AdaBoost_rand_clf_params = AdaBoost_rand_clf.best_params_
    
    print(AdaBoost_rand_clf_params) 
   
    
    ###construct an AdaBoost classifier using grid search for hyperparameter tuning
    
    AdaBoost_parameters = {'n_estimators': [70,75,80], 'learning_rate':[0.1,0.15,0.2], 'algorithm': ['SAMME'], 'random_state': [0]}
    
    AdaBoost_clf = GridSearchCV(AdaBoostClassifier(), param_grid = AdaBoost_parameters, scoring = scoring, cv=folds)
    AdaBoost_clf.fit(X,y)
    
    #identify the best AdaBoost classifier in grid search
    best_AdaBoost_clf = AdaBoost_clf.best_estimator_
    
    #identify the most important factor in the best Random Forest classifier
    Ada_features = extract_factors(X.columns, best_AdaBoost_clf)
    print(Ada_features)
    
    #Calculate model diagnostics of Best AdaBoost Classifier
    
    Ada_diagnostics = model_diagnostics(best_AdaBoost_clf, X, y)
    print(Ada_diagnostics.summary)
    
    '''
    Search for the most important features using Gradient Boost
    '''
    
    ###construct an GradientBoost classifier using random search for ball parking good hyperparameters with respect to Precison Recall Curve as selection criteria
    
    GradientBoost_Rand_parameters = {'loss' : ['deviance', 'exponential'],'learning_rate':[0.05,0.1,0.2,0.3,0.5,0.75,1],'n_estimators':[10,20,50,75,100,125,150,175], 'min_samples_leaf': [1,2,4], 'warm_start': [True,False], 'min_samples_split': [2,5,10], 'max_depth': [1,2,3,5,8,10,20,40,70],'criterion' : ['friedman_mse','mse'], 'random_state':[66]}
    
    GradientBoost_rand_clf = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions = GradientBoost_Rand_parameters, cv = folds, verbose = 2, scoring = scoring, n_iter=300)
    GradientBoost_rand_clf.fit(X,y)
    
    GradientBoost_rand_clf_params = GradientBoost_rand_clf.best_params_
    
    print(GradientBoost_rand_clf_params) 
    
    
    ###construct a gradient boost classifier
    
    GradientBoost_parameters = {'n_estimators': [45,50,55], 'loss': ['deviance'], 'learning_rate': [0.05,0.1,0.15], 'max_depth':[1,2,3], 'warm_start': [True], 'min_samples_leaf': [2],'min_samples_split': [10], 'random_state':[0]}
    
    
    GradientBoost_clf = GridSearchCV(GradientBoostingClassifier(), param_grid = GradientBoost_parameters, scoring = scoring, cv= folds)
    GradientBoost_clf.fit(X,y)
    
    best_GradientBoost_clf = GradientBoost_clf.best_estimator_
    
    #identify important features
    GradientBoost_features = extract_factors(X.columns, best_GradientBoost_clf)
    print(GradientBoost_features)
    
    #Calculate model diagnostics of Best Gradient Boost Classifier
    
    GB_diagnostics = model_diagnostics(best_GradientBoost_clf, X, y)
    
    print(GB_diagnostics.summary)
    
    '''
    For Prediction/Predictive Modeling, a voting classifier will be employed.
    This is strictly for predictive purposes only.
    '''
    
    ###construct voting classifier with all 3 models to identify if jointly the models perform better in prediction
    
    clfs = [('Ada', best_AdaBoost_clf),('RF', best_RF_clf), ('GB', best_GradientBoost_clf)]
    
    CV_params = {'weights':[[1,2,3],[1,4,9]]}
    
    GridSearch_Vote_clf = GridSearchCV(VotingClassifier(clfs, voting = 'soft'), param_grid = CV_params, scoring = 'precision', cv = folds)
    GridSearch_Vote_clf.fit(X,y)
    
    
    Vote_clf = GridSearch_Vote_clf.best_estimator_
    Vote_clf.fit(X,y)
    
    ###calculate model diagnostics for voting classifier
    Vote_diagnostics = model_diagnostics(Vote_clf, X, y)
    
    print(Vote_diagnostics.summary)
    
    