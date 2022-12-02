# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:36:40 2022

@author: luis_
"""

# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
import multiprocessing

# Importing the dataset
dataset = pd.read_csv('C:/Users/luis_/Estancia de investigación/DatabaseFinal.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


############################################################################################
#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#########################################################################################


# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly', degree = 4, gamma = 'scale', coef0 = 0.0, C = 1)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

#Error cuadratico medio
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred)
print("Error cuadratico medio MSE: ", MSE)

#Error porcentual absoluto
from sklearn.metrics import mean_absolute_percentage_error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("Error porcentual absoluto MAPE: ", MAPE)

#Factor R2
from sklearn.metrics import median_absolute_error
MedAE = median_absolute_error(y_test, y_pred)
print("Error absoluto medio MedAE: ", MedAE)


#Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accurracies = cross_val_score(estimator = regressor, X = X_train, y = y_train,scoring='neg_mean_squared_error', cv = 10)
print("Error cuadratico medio Cross Validation Media: ",accurracies.mean())
print("Error cuadratico medio Cross Validation STD: ", accurracies.std())



importancia = permutation_importance(
                estimator    = regressor,
                X            = X_test,
                y            = y_test,
                n_repeats    = 5,
                scoring      = 'neg_root_mean_squared_error',
                n_jobs       = multiprocessing.cpu_count() - 1,
                random_state = 0
             )


# Se almacenan los resultados (media y desviación) en un dataframe
df_importancia = pd.DataFrame(
                    {k: importancia[k] for k in ['importances_mean', 'importances_std']}
                 )
df_importancia['feature'] = ['Red','Green','Blue','Depth']
df_importancia.sort_values('importances_mean', ascending=False)



# Gráfico
fig, ax = plt.subplots(figsize=(5, 6))
df_importancia = df_importancia.sort_values('importances_mean', ascending=True)
ax.barh(
    df_importancia['feature'],
    df_importancia['importances_mean'],
    xerr=df_importancia['importances_std'],
    align='center',
    alpha=0
)
ax.plot(
    df_importancia['importances_mean'],
    df_importancia['feature'],
    marker="D",
    linestyle="",
    alpha=0.8,
    color="r"
)
ax.set_title('Predictors importance (train)')
ax.set_xlabel('Error increment after permutation');



# #Aplicar Grid Search
# from sklearn.model_selection import GridSearchCV
# parameters = [{'kernel':['poly'], 'degree':[1,2,3,4,5,6],'gamma':['scale'] ,'coef0':[0.0,0.1,0.2],'C':[1,2,3]}]


# grid_search = GridSearchCV(estimator = regressor,
#                            param_grid = parameters,
#                            scoring ='neg_mean_squared_error',
#                            cv = 10,
#                            n_jobs = -1)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_

from joblib import dump
dump(regressor, 'SupportVectorMachine.joblib') 

