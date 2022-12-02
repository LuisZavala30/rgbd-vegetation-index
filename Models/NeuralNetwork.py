# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:38:48 2022

@author: luis_
"""

# Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import shap
import multiprocessing

dataset = pd.read_csv('###INSERT PATH OF THE DATABASE####')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Construir NN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Inicializar la RNA
regressor = Sequential()

#Añadir capas de entrada y capas ocultas
regressor.add(Dense(units = 4, kernel_initializer = "uniform", activation = "relu", input_dim = 4))
regressor.add(Dense(units = 4, kernel_initializer = "uniform", activation = "relu"))

#Añadir la capa de salida
regressor.add(Dense(units = 1, kernel_initializer = "uniform", activation = "tanh"))

# Compilar red neuronal
regressor.compile(optimizer="adam", loss = "mean_squared_error", metrics = ['mae','mse'])

#Ajustar red neuronal al conjunto de entrenamiento
regressor.fit(X_train, y_train, batch_size = 5, epochs = 500)


y_pred = regressor.predict(X_test)

regressor.save('NDVIEstimator.h5')


#METRICAS DE EVALUACIÓN
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

DE = shap.DeepExplainer(regressor, X_train) # X_train is 3d numpy.ndarray
shap_values = DE.shap_values(X_test, check_additivity=False) # X_validate is 3d numpy.ndarray

shap.initjs()
shap.summary_plot(
     shap_values[0], 
     X_test,
     feature_names=['Red','Green','Blue','Depth'],
     max_display=50,
     plot_type='bar')






