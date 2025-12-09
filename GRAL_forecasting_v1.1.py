#libraries
#from lightgbm.plotting import _determine_direction_for_categorical_split
import numpy as np
import pandas as pd
import os
#from datetime import timedelta

import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')

import warnings
warnings.simplefilter("ignore")

#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import StandardScaler
#from tkinter import *

#import lightgbm as lgb

#from skforecast.recursive import ForecasterRecursive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from utils.metrics import evaluate
import pickle

resultDict={}
predictionDict={}


#data set with the input data
def read_data_sets(fileName1, fileName2):
    raw_prices_data = pd.read_csv(fileName1, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    mat_prices_data = pd.read_csv(fileName2, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    raw_prices_data['Date']=pd.to_datetime(raw_prices_data['Date'], format='mixed',dayfirst=True) 
    mat_prices_data['Date']=pd.to_datetime(mat_prices_data['Date'], format='mixed',dayfirst=True) 
    return raw_prices_data, mat_prices_data



def display_plot(predictions_df, model_type,filename):
        
        plt.figure(figsize=(10,5))
        plt.plot(predictions_df['Date'].tail(12), predictions_df['Prediction'].tail(12), color='grey', label=model_type)
        plt.plot(predictions_df['Date'].tail(12), predictions_df['Actual'].tail(12), label='Original', color="blue")       
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        file_path = os.path.join(BASE_DIR, "output", filename)
        plt.savefig(file_path)
        plt.show()

def predict_raw_prices_model(filename, model_type='xgb', split_date='2020-01-01'):

    prices_data = pd.read_csv(filename, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    prices_data['Date']=pd.to_datetime(prices_data['Date'], format='mixed',dayfirst=True)

    split_date = pd.to_datetime(split_date, dayfirst=True)
    df_training = prices_data.loc[prices_data['Date'] <= split_date]
    df_test = prices_data.loc[prices_data['Date'] > split_date]

    print(f"{len(df_training)} months of the training data {len(df_test)} months of testing data")

    df_training.to_csv('output/training.csv')
    df_test.to_csv('output/test.csv')

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

    X_train_df, y_train = create_time_features(df_training, target='column_1')
    X_test_df, y_test = create_time_features(df_test, target='column_1')

    scaler = StandardScaler()
    scaler.fit(X_train_df)  # No cheating, never scale on the training+test!
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
    X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns)

    if model_type == 'xgb':
        model = XGBRegressor(random_state=15926)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(model.get_params())
        
        resultDict['XGBoost'] = evaluate(df_test.column_1, yhat)
        predictionDict['XGBoost'] = yhat
        
       
       
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=15926)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(model.get_params())
        
        resultDict['Randomforest'] = evaluate(df_test.column_1, yhat)
        predictionDict['Randomforest'] = yhat
        depths = [estimator.tree_.max_depth for estimator in model.estimators_]
        print(np.mean(depths))
       
    elif model_type == 'lgbm':
        model = LGBMRegressor(min_data_in_leaf=1, min_data_in_bin=1)
    else:
        raise ValueError("Unsupported model")

    
    predictions_df = pd.DataFrame({
        'Date': df_test['Date'],
        'Actual': df_test['column_1'].values,
        'Prediction': yhat
    })

    display_plot(predictions_df, model_type, 'raw_material_prices.png')
    return predictions_df

    
from statsmodels.tsa.arima.model import ARIMA
def predict_raw_prices_ARIMA(data):
    model=ARIMA(data, order=(3,1,1))
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=10)

    return model_fit, prediction



from statsmodels.tsa.holtwinters import ExponentialSmoothing
def predict_raw_prices_HoltWinters(data):
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=10)
   
    return model_fit, prediction


def correlation(merged_prices, raw_materials_col_name, supplier_material_col_name):   
    corr = merged_prices[raw_materials_col_name].corr(merged_prices[supplier_material_col_name], method='pearson')
    print ("Correlation between ", raw_materials_col_name, " and ", supplier_material_col_name, "is: ", round(corr, 2))
    if (abs(corr)<0.1):
        correlation=False
    else:
        correlation = True
    return correlation

def linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name, months_prediction, model_type):
    # Splitting variables
    X = merged_prices[raw_materials_col_name].values.reshape(-1, 1)
    y = merged_prices[supplier_material_col_name].values.reshape(-1, 1)
    
    # Splitting dataset into test/train
    X_train, X_test = X[:-months_prediction], X[-months_prediction:]
    y_train, y_test = y[:-months_prediction], y[-months_prediction:]

    # Regressor model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Prediction result
    y_pred_test = regressor.predict(X_test)     # predicted value of y_test        
    y_full_pred = np.concatenate([y_train, y_pred_test])
    y_full_real = np.concatenate([y_train, y_test])  

    
    resultDict[model_type] = evaluate(y_test, y_pred_test)
    predictionDict[model_type] = y_pred_test

    predictions_df = pd.DataFrame({
        'Date': merged_prices['Date'],
        'Actual': y_full_real.flatten(),
        'Prediction': y_full_pred.flatten()
    })

    display_plot(predictions_df, 'linear_'+model_type, 'predictions.png')
    return predictions_df

def linear_model_delays(merged_prices, raw_materials_col_name, supplier_material_col_name,
                 months_prediction, model_type, n_lags=3):
   

    df = merged_prices.copy()

    # Crear retardos del crudo
    for lag in range(1, n_lags + 1):
        df[f"{raw_materials_col_name}_lag{lag}"] = df[raw_materials_col_name].shift(lag)

    # Eliminar filas con NaN generados por los lags
    df = df.dropna().reset_index(drop=True)

    # Features (X) y target (y)
    feature_cols = [raw_materials_col_name] + [f"{raw_materials_col_name}_lag{lag}" for lag in range(1, n_lags + 1)]
    X = df[feature_cols].values
    y = df[supplier_material_col_name].values.reshape(-1, 1)

    # Train/test split
    X_train, X_test = X[:-months_prediction], X[-months_prediction:]
    y_train, y_test = y[:-months_prediction], y[-months_prediction:]

    # Modelo lineal
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicciones
    y_pred_test = regressor.predict(X_test)
    y_full_pred = np.concatenate([regressor.predict(X_train), y_pred_test])
    y_full_real = np.concatenate([y_train, y_test])

    resultDict[model_type] = evaluate(y_test, y_pred_test)
    predictionDict[model_type] = y_pred_test

    # DataFrame para graficar
    predictions_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual': y_full_real.flatten(),
        'Prediction': y_full_pred.flatten()
    })

    display_plot(predictions_df, 'linear_'+model_type)
    return predictions_df

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
def SMPP_RF(fileName1, fileName2):
    #fileName1 = csv file containing the dataset with the raw material prices
    #fileName2= csv file containing the dataset with the raw material prices

   #  raw_materials_col_name='column_1'      #name of the column in the raw materials dataset
   #  raw_materials_col_name2='column_2'      #name of a second columnd in the raw materials dataset
   #  supplier_material_col_name='column'    #name of the column with the prices in the suppliers material dataset    
   #  x_axis_text='Raw material prices ($/unit)'        #name of the axis for plotting the raw material prices datasets 
       
   #  #read the datasets
   # # raw_prices_data, mat_prices_data= read_data_sets(fileName1, fileName2)
   #  raw_prices_data=raw_prices_data[['Date', 'column_1']]
   
   #  forecaster, predictionXGBoost=predict_raw_prices_model(fileName1, model_type='xgb')

    
   #  merged_prices = pd.merge(mat_prices_data,raw_prices_data,left_on='Date',right_on='Date')  
   #  corr=correlation(merged_prices,raw_materials_col_name, supplier_material_col_name)

   #  if(corr):
   #      #find the linear regression model
   #      reg, score=linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name)  
    
   #      X_prediction = predictionXGBoost.values.reshape(-1,1)
   #      y = reg.predict(X_prediction) 
        
   #      #formating the ouptut to contain the date, the raw marterial predicted prices (RMPP) and the supplier material predicted prices (SMOO)
   #      d = {'Date':predictionXGBoost.index, 'RMPP':predictionXGBoost.values,'SMPP':y.flatten()}
   #      dfData = pd.DataFrame(data=d) 
   #      print(corr)
   #  else:
   #      sys.exit()
        
   #  return dfData[['Date', 'RMPP', 'SMPP']], merged_prices
   return fileName1



def SMPP_LightBM(fileName1, fileName2):
    #fileName1 = csv file containing the dataset with the raw material prices
    #fileName2= csv file containing the dataset with the raw material prices

    raw_materials_col_name='column_1'      #name of the column in the raw materials dataset
    raw_materials_col_name2='column_2'      #name of a second columnd in the raw materials dataset
    supplier_material_col_name='column'    #name of the column with the prices in the suppliers material dataset    
    x_axis_text='Raw material prices ($/unit)'        #name of the axis for plotting the raw material prices datasets 
       
    #read the datasets
    raw_prices_data, mat_prices_data= read_data_sets(fileName1, fileName2)
    raw_prices_data=raw_prices_data[['Date', 'column_1']]
    #raw_prices_data['column_1'] = raw_prices_data['column_1'] + np.random.normal(0, 5, len(raw_prices_data))

    #Joins the prices dataset by the date to remove non-existing values.
    merged_prices = pd.merge(mat_prices_data,raw_prices_data,left_on='Date',right_on='Date')    
    
    #prediction moodel in forecaster and predicted values in prediction
    forecaster, predictionLightbm=predict_raw_prices_model(fileName1, model_type='lgbm')  

    corr=correlation(merged_prices,raw_materials_col_name, supplier_material_col_name)

    if(corr):
        #find the linear regression model
        reg, score=linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name)  
    
        X_prediction = predictionLightbm.values.reshape(-1, 1)
    
        #predict the values
        y = reg.predict(X_prediction) 
        
        #formating the ouptut to contain the date, the raw marterial predicted prices (RMPP) and the supplier material predicted prices (SMOO)
        d = {'Date':predictionLightbm.index, 'RMPP':predictionLightbm.values,'SMPP':y.flatten()}
        dfData = pd.DataFrame(data=d) 
        print(corr)
    else:
        sys.exit()
        
    return dfData[['Date', 'RMPP', 'SMPP']], merged_prices


def SMPP_XGBoost(fileName1, fileName2, split_date):
    #fileName1 = csv file containing the dataset with the raw material prices
    #fileName2= csv file containing the dataset with the raw material prices

    raw_materials_col_name='column_1'      #name of the column in the raw materials dataset
    #raw_materials_col_name2='column_2'      #name of a second columnd in the raw materials dataset
    supplier_material_col_name='column'    #name of the column with the prices in the suppliers material dataset    
    #x_axis_text='Raw material prices ($/unit)'        #name of the axis for plotting the raw material prices datasets 
       
    #read the datasets
    raw_prices_data = pd.read_csv(fileName1, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    raw_prices_data['Date']=pd.to_datetime(raw_prices_data['Date'], format='mixed',dayfirst=True)

    mat_prices_data = pd.read_csv(fileName2, delimiter=';',parse_dates=["Date"], date_format='%Y-%m-%d')
    mat_prices_data['Date']=pd.to_datetime(mat_prices_data['Date'], format='mixed',dayfirst=True)

        
    merged_prices = pd.merge(mat_prices_data,raw_prices_data,left_on='Date',right_on='Date')
    #data =format_raw_materials_price(raw_prices_data, raw_materials_col_name)
    predictions_df=predict_raw_prices_model(fileName1, model_type='xgb', split_date=split_date)   

    corr=correlation(merged_prices,raw_materials_col_name, supplier_material_col_name)

    if(corr):
        #find the linear regression model
        prediction_df_linear=linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name, 12, model_type='xgb')
        
        #formating the ouptut to contain the date, the raw marterial predicted prices (RMPP) and the supplier material predicted prices (SMOO)
        print(type( prediction_df_linear))
        d = {'Date':prediction_df_linear['Date'], 'real pbt':prediction_df_linear['Actual'],'predicted PBT': prediction_df_linear['Prediction']}        
        dfData = pd.DataFrame(data=d) 
      #  print(corr)
    else:
        sys.exit()
    print (dfData)          
    return dfData, merged_prices

def SMPP_ARIMA(fileName1, fileName2):
    #fileName1 = csv file containing the dataset with the raw material prices
    #fileName2= csv file containing the dataset with the raw material prices

    # raw_materials_col_name='column_1'      #name of the column in the raw materials dataset
    # raw_materials_col_name2='column_2'      #name of a second columnd in the raw materials dataset
    # supplier_material_col_name='column'    #name of the column with the prices in the suppliers material dataset    
    # x_axis_text='Raw material prices ($/unit)'        #name of the axis for plotting the raw material prices datasets 
       
    # #read the datasets
    # #raw_prices_data, mat_prices_data= read_data_sets(fileName1, fileName2)
    # data =format_raw_materials_price(raw_prices_data, raw_materials_col_name)

    # #Joins the prices dataset by the date to remove non-existing values.
    # merged_prices = pd.merge(mat_prices_data,raw_prices_data,left_on='Date',right_on='Date')
    # forecaster, prediction=predict_raw_prices_ARIMA(data)
      
    
    # corr=correlation(merged_prices,raw_materials_col_name, supplier_material_col_name)
    # if(corr):
    #     #find the linear regression model
    #     reg, score=linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name)  
    
    #     X_prediction = prediction.values.reshape(-1, 1)      
    #     y = reg.predict(X_prediction) 
        
    #     #formating the ouptut to contain the date, the raw marterial predicted prices (RMPP) and the supplier material predicted prices (SMOO)
    #     d = {'Date':prediction['Date'], 'RMPP':prediction,'SMPP':y.flatten()}
    #     dfData = pd.DataFrame(data=d) 
    #     print(corr)
    # else:
    #     sys.exit()
        
    # return dfData, merged_prices
    return fileName1

def SMPP_HoltWinter(fileName1, fileName2):

    return fileName1
   
    # raw_materials_col_name='column_1'      #name of the column in the raw materials dataset
    # raw_materials_col_name2='column_2'      #name of a second columnd in the raw materials dataset
    # supplier_material_col_name='column'    #name of the column with the prices in the suppliers material dataset    
    # x_axis_text='Raw material prices ($/unit)'        #name of the axis for plotting the raw material prices datasets 
       
    # #read the datasets
    # #raw_prices_data, mat_prices_data= read_data_sets(fileName1, fileName2)
    # data =format_raw_materials_price(raw_prices_data, raw_materials_col_name)

    # merged_prices = pd.merge(mat_prices_data,raw_prices_data,left_on='Date',right_on='Date')  
    # forecaster, prediction=predict_raw_prices_HoltWinters(data)

    # corr=correlation(merged_prices,raw_materials_col_name, supplier_material_col_name)
    # if(corr):
    #     #find the linear regression model
    #     reg, score=linear_model(merged_prices, raw_materials_col_name, supplier_material_col_name)  
    
    #     X_prediction = prediction.values.reshape(-1, 1)
    
    #     #predict the values
    #     y = reg.predict(X_prediction) 
        
    #     #formating the ouptut to contain the date, the raw marterial predicted prices (RMPP) and the supplier material predicted prices (SMOO)
    #     d = {'Date':prediction.index, 'RMPP':prediction,'SMPP':y.flatten()}
    #     dfData = pd.DataFrame(data=d) 
    # else:
    #     sys.exit()
    # print(dfData)
    # return dfData, merged_prices

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


if __name__ == "__main__":
   
    if len(sys.argv) < 3:
         print("Uso: python GRAL_forecasting_v1.1.py <archivo1> <archivo2>")
         print(f"Error, forcing reading default data")
         fileName1="input/Oil_original.csv"
         fileName2="input/PBT_historical_prices.csv"
    else:
        fileName1=sys.argv[1]
        fileName2=sys.argv[2]
   
    split_date='2024-07-01'

    # dfPricesXGBoost, mergedPricesXGBoost = SMPP_XGBoost(fileName1, fileName2)    
    # #print("The amount of variability of the supplier prices that is explained by the raw material prices is %.2f" % score)
    # mergedPrices=mergedPricesXGBoost.rename(columns={"column_1": "RMPP", "column": "SMPP"})
    # df=pd.concat([mergedPrices, dfPricesXGBoost])
    # df = df.reset_index(drop=True)
    # file_path1 = os.path.join(BASE_DIR, "output", "forecastXGBoost.csv")
    # df.to_csv(file_path1)
    # plot_result(df)

    # dfPricesLightBM, mergedPricesLightBM =SMPP_LightBM(fileName1, fileName2)    
    # #print("The amount of variability of the supplier prices that is explained by the raw material prices is %.2f" % score)
    # mergedPrices=mergedPricesLightBM.rename(columns={"column_1": "RMPP", "column": "SMPP"})
    # df2=pd.concat([mergedPrices, dfPricesLightBM])
    # df2 = df2.reset_index(drop=True)
    # file_path2 = os.path.join(BASE_DIR, "output", "forecastLightBM.csv")
    # df2.to_csv(file_path2)
    # plot_result(df2)


    dfPricesRF, mergedPricesRF = SMPP_XGBoost(fileName1, fileName2, split_date)    
    #print("The amount of variability of the supplier prices that is explained by the raw material prices is %.2f" % score)
    # mergedPrices=mergedPricesRF.rename(columns={"column_1": "RMPP", "column": "SMPP"})
    # df3=pd.concat([mergedPrices, dfPricesRF])
    # df3 = df3.reset_index(drop=True)
    file_path2 = os.path.join(BASE_DIR, "output", "forecastXGBoost.csv")
    dfPricesRF.to_csv(file_path2)
    #plot_result(df3)

    # dfPricesARIMA, mergedPricesARIMA = SMPP_ARIMA(fileName1, fileName2)    
    # #print("The amount of variability of the supplier prices that is explained by the raw material prices is %.2f" % score)
    # mergedPrices=mergedPricesARIMA.rename(columns={"column_1": "RMPP", "column": "SMPP"})
    # df=pd.concat([mergedPrices, dfPricesARIMA])
    # df = df.reset_index(drop=True)
    # file_path1 = os.path.join(BASE_DIR, "output", "forecastARIMA.csv")
    # df.to_csv(file_path1)
    # plot_result(df)

    # dfPricesHoltWinter, mergedPricesHoltWinter = SMPP_HoltWinter(fileName1, fileName2)    
    # #print("The amount of variability of the supplier prices that is explained by the raw material prices is %.2f" % score)
    # mergedPrices=mergedPricesHoltWinter.rename(columns={"column_1": "RMPP", "column": "SMPP"})
    # df=pd.concat([mergedPrices, dfPricesHoltWinter])
    # df = df.reset_index(drop=True)
    # file_path1 = os.path.join(BASE_DIR, "output", "forecastHoltWinter.csv")
    # df.to_csv(file_path1)
    # plot_result(df)

    with open('output/scores.pickle', 'wb') as handle:
        pickle.dump(resultDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('output/predictions.pickle', 'wb') as handle:
        pickle.dump(predictionDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('output/scores.pickle', 'rb') as handle:
        resultsDict = pickle.load(handle)
        print(resultsDict)
