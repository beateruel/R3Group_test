# %% [markdown]
# ## Price forecasting using XGBoost: tested with GLN plast/ GOR

# The script contains the functions for estimating the 1st suppliers materials by using the estimation of raw material prices.
# Assumptions:
# * Raw materials price dataset available is large. It allows using advanced machine learning techniques.
# * Supplier's material price dataset is small.
# 
# Then the following steps are followed:
# * 1) Find the XGboost model from the raw material dataset available.
# * 2) Calculate the prediction error by using the dataset available.
# * 3) Assuming the error is acceptable, we estimate the future values.
# * 4) Calculate the linear correlation between the raw material prices and the Supplier material prices.
# * 5) Assuming the correlation is over 0.29, we find the linear regression with the datasets available.
# * 6) Calculate the prediction error by using the dataset available.
# * 7) Assuming the estimation error is acceptable, we calculate the supplier material prices by using the linear regression model and the raw material predicted values.
#
#libraries
import time
from datetime import date
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.simplefilter("ignore")
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from utils.metrics import evaluate
from tkinter import *
from tkinter import filedialog


# Modelling and Forecasting
# ==============================================================================

import skforecast
import sklearn
from xgboost import XGBRegressor
#from skforecast.feature_selection  import select_features
from sklearn.feature_selection import RFECV
from skforecast.recursive import ForecasterRecursive
#from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster

#from skforecast.datasets import fetch_dataset


# Warnings configuration

#https://skforecast.org/0.13.0/user_guides/autoregresive-forecaster.html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def create_time_features(df, target=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['sin_day'] = np.sin(df['dayofyear'])
    df['cos_day'] = np.cos(df['dayofyear'])
    df['dayofmonth'] = df['Date'].dt.day
 
    X = df.drop(['Date'], axis=1)
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y

    return X

#data set with the input data
def read_data_sets(fileName1, fileName2):
    raw_prices_data = pd.read_csv(fileName1, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    mat_prices_data = pd.read_csv(fileName2, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    raw_prices_data['Date']=pd.to_datetime(raw_prices_data['Date'], format='mixed',dayfirst=True) 
    mat_prices_data['Date']=pd.to_datetime(mat_prices_data['Date'], format='mixed',dayfirst=True) 

    return raw_prices_data, mat_prices_data


def split_materials_price(raw_prices_data, raw_materials_col_name):
    #To facilitate the training of the models, the search for optimal hyperparameters and the evaluation of their predictive accuracy, 
    #the data are divided into three separate sets: training, validation and test.
    # Split train-validation-test

      # ==============================================================================
    data =raw_prices_data
    # Data preprocessing 
    # ==============================================================================
    data['datetime'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.set_index('Date')
    data = data.asfreq('MS')
    data= data[raw_materials_col_name]
    data = data.sort_index()

    #Splitting data
    # ==============================================================================
    end_train = int(len(data)*0.6) #60% of the data
    end_validation = int(len(data)*0.85) #25% of the data, the remaining 15% for the test set
    data_train = data[: end_train]
    data_val   = data[end_train:end_validation]
    data_test  = data[end_validation:]

    print(f"Dates train      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Dates validacion : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
    print(f"Dates test       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")
    
    return data_train, data_val, data_test,data




# #### Data exploration
# Graphical exploration of time series can be an effective way of identifying trends, patterns, and seasonal variations. This, in turn, helps to guide the selection of the most appropriate forecasting model.


def plot_splitted_data(data_train, data_val, data_test, x_axis_text):
    # Interactive plot of time series
    # ==============================================================================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_train.index, y=data_train, mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=data_val.index, y=data_val, mode='lines', name='Validation'))
    fig.add_trace(go.Scatter(x=data_test.index, y=data_test, mode='lines', name='Test'))
    fig.update_layout(
        title  = x_axis_text,
        xaxis_title="Time",
        yaxis_title="Price",
        legend_title="Partition:",
        width=800,
        height=350,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.001
        )
    )
    #fig.update_xaxes(rangeslider_visible=True)
   # fig.show()
    return 1

def predict_raw_prices(data):
    # Create forecaster
    # ==============================================================================
    forecaster = ForecasterRecursive(
                     regressor = XGBRegressor(random_state=15926, enable_categorical=True),
                     lags      = 24
                 )
    
    # Train forecaster
    # ==============================================================================
    end_validation = int(len(data)*0.85) #25% of the data, the remaining 15% for the test set
    forecaster.fit(y=data[end_validation:])

    #Predict
    #================================================================================
    prediction=forecaster.predict(steps=10)

    return forecaster, prediction


def correlation(merged_prices, raw_materials_col_name, supplier_material_col_name):

    corr = merged_prices[raw_materials_col_name].corr(merged_prices[supplier_material_col_name], method='pearson')
    print ("Correlation between ", raw_materials_col_name, " and ", supplier_material_col_name, "is: ", round(corr, 2))
    if (abs(corr)<0.1):
       # print("it presents a low correlation and we can not continous with the forecast")
        correlation=False
    else:
      #  print("it presents a high correlation and we can continous with the forecast")
        correlation = True

    print(correlation)

    return correlation


def linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name):
    # Splitting variables
    X = merged_prices[raw_materials_col_name].values.reshape(-1, 1)
    y = merged_prices[supplier_material_col_name].values.reshape(-1, 1)
    # Splitting dataset into test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # Regressor model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Prediction result
    y_pred_test = regressor.predict(X_test)     # predicted value of y_test
    #print(y_test)
    #print(y_pred_test)

    score = regressor.score(X, y) #Return the coefficient of determination of the prediction.


    return regressor,score

def plot_result(df):
    # Create some mock data
    t = df.Date
    data1 = df.RMPP
    data2 = df.SMPP
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Raw material', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Supplier Material', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    file_path = os.path.join(BASE_DIR, "output", "predictions.png")
    plt.savefig(file_path)
    graph=plt.show()
   
    return graph

import sys
def main(fileName1, fileName2):
    #fileName1 = csv file containing the dataset with the raw material prices
    #fileName2= csv file containing the dataset with the raw material prices

    raw_materials_col_name='column_1'      #name of the column in the raw materials dataset
    raw_materials_col_name2='column_2'      #name of a second columnd in the raw materials dataset
    supplier_material_col_name='column'    #name of the column with the prices in the suppliers material dataset    
    x_axis_text='Raw material prices ($/unit)'        #name of the axis for plotting the raw material prices datasets 

    
    #read the datasets
    raw_prices_data, mat_prices_data= read_data_sets(fileName1, fileName2)
    print(fileName1)

    #Joins the prices dataset by the date to remove non-existing values.
    merged_prices = pd.merge(mat_prices_data,raw_prices_data,left_on='Date',right_on='Date')
    print(merged_prices)
   
    #raw_materials_col_name = name of the column in the dataframe containing the raw material prices
    #raw_materials_col_name2 = name of the column in the dataframe containing additional values 

    #function to train and test the model to forecast the raw_material_prices
    data_train, data_val, data_test,data =split_materials_price(raw_prices_data, raw_materials_col_name)
    
    #x_axis_text = text will be display in the graph
    #graph plotting the data in each range of data
    plot_splitted_data(data_train, data_val, data_test, x_axis_text)

    #prediction moodel in forecaster and predicted values in prediction
    forecaster, prediction=predict_raw_prices(data)

    # Backtest model on test data
    #https://skforecast.org/0.13.0/user_guides/backtesting
    # ==============================================================================
    end_validation = int(len(data)*0.85) #25% of the data, the remaining 15% for the test set


   # metric, predictions = backtesting_forecaster(
                         # forecaster         = forecaster,
                         # y                  = data,
                          #fh                 = 36,
                          #metric             = 'mean_absolute_percentage_error',
                          #initial_train_size = len(data[:end_validation]),
                          #refit              = False,
                          #n_jobs             = 'auto',
                          #verbose            = False, # Change to False to see less information
                          #show_progress      = True
                   #   )


    # Backtesting error
    # ==============================================================================
    #print(metric)
    #supplier_material_col_name = name of the column in the dataframe containing the supplier material prices
    
    #star with the correlation and linear regression model between two diferent datasets.
    corr=correlation(merged_prices,raw_materials_col_name, supplier_material_col_name)

    if(corr):
        #find the linear regression model
        reg, score=linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name)  
    
        X_prediction = prediction.values.reshape(-1, 1)
    
        #predict the values
        y = reg.predict(X_prediction) 
        
        #formating the ouptut to contain the date, the raw marterial predicted prices (RMPP) and the supplier material predicted prices (SMOO)
        d = {'Date':prediction.index, 'RMPP':prediction,'SMPP':y.flatten()}
        dfData = pd.DataFrame(data=d) 
        print(corr)
    else:
        sys.exit()
        
    return dfData, merged_prices
       

if __name__ == "__main__":
   
    if len(sys.argv) < 3:
         print("Uso: python GRAL_forecasting_v1.1.py <archivo1> <archivo2>")
         sys.exit(1)

    

    fileName1=sys.argv[1]
    fileName2=sys.argv[2]
    print(f"Error: {sys.argv[1]}")

    dfPrices, mergedPrices = main(fileName1, fileName2)
    
    #print("The amount of variability of the supplier prices that is explained by the raw material prices is %.2f" % score)
    mergedPrices=mergedPrices.rename(columns={"column_1": "RMPP", "column": "SMPP"})
    df=pd.concat([mergedPrices, dfPrices])
    df = df.reset_index(drop=True)
    file_path = os.path.join(BASE_DIR, "output", "forecast.csv")
    df.to_csv(file_path)
    plot_result(df)

