# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:48:07 2025

@author: Muhammad Adil Saleem

Purpose: Generation Forecast Error modelling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
# figure out current working directory,
# Change the wd to the folder of technical challenge "Extract first if zipped"
cwd = os.getcwd()
print("Current working directory:", cwd)
#if needed, change it to desired wd
os.chdir(r'C:\Users\user\Enoda-forecasting')

# importing data as a .csv file from current working directory
file_name= 'data for modelling.csv'
df = pd.read_csv(file_name)
print(df.head())

# since it is the time series analysis, first we need to set time as index
# Set 'time' as the index
df.set_index('time', inplace=True)

#let inspect Generation Forecast Error (GFE) Graphically
# Assuming 'df' is your DataFrame with a datetime index and 'variable_column' is the variable to plot
subsample = df.loc['2024-01-01':'2025-02-20']  # Filter by date range

# Plot the subsample
subsample.plot(y='GFE', kind='line')  # No need to specify x as the index is used
plt.title('Generation forecast error')
plt.show()

# Before starting with time series data let's check for the stationary through
# Augmented Dicky Fuller (ADF) test for selected_variables time series variables.

selected_variables = df[['GFE', 'LFE', 'EDP']]
# filling Nan with 0 (The first observation of series)
df_filled = df.fillna(0)
# Function to perform ADF test and interpret results
def test_stationarity(series, name):
    result = adfuller(series)
    print(f'ADF Test for {name}:')
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    # Compare test statistic with critical values
    if result[0] < result[4]['5%']:
        print(f'Result: {name} is stationary (reject null hypothesis)\n')
    else:
        print(f'Result: {name} is non-stationary (fail to reject null hypothesis)\n')

# Test stationarity for the selected variables
for var in selected_variables:
    test_stationarity(df_filled[var], var)
    
#Since all our three variables are stationary, its safe to use them without differencing

# """""""""""function for exporting model summary as an image
def export_results_as_image(model, filename):
    """
    Exports regression results as an image (PNG) with values rounded to 4 decimal places.
    Includes the filename as the title of the table.
    
    Parameters:
        model: Fitted statsmodels regression model.
        filename: Name of the output image file (e.g., 'results.png').
    """
    # Extract coefficients and other statistics
    coeff_table = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values.round(4),  # Round to 4 decimal places
        'Std Error': model.bse.values.round(4),      
        't-value': model.tvalues.values.round(4),    
        'p-value': model.pvalues.values.round(4)     
    })

    # adding other statistics to the table (rounded to 4 decimal places)
    other_stats = pd.DataFrame({
        'Statistic': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'AIC', 'BIC'],
        'Value': [
            round(model.rsquared, 4),
            round(model.rsquared_adj, 4),
            round(model.fvalue, 4),         
            round(model.f_pvalue, 4),       
            round(model.aic, 4),            
            round(model.bic, 4)           
        ]
    })

    # Combine tables for visualization
    results_table = pd.concat([coeff_table, other_stats], ignore_index=True)

    # Plot the table as an image
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size to accommodate title
    ax.axis('off')  # Hide axes

    # Add title (filename without extension)
    title = filename.split('.')[0]  # Remove file extension
    plt.title(title, fontsize=14, pad=20)  # Add title with padding

    # Create table
    table = ax.table(
        cellText=results_table.values,
        colLabels=results_table.columns,
        cellLoc='center',
        loc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the table as an image
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
#"""""""" function closed"""""""

# Function for regression diagnostics (testing for heteroscedasticity and autocorrelation)
def regression_diagnostics(model):
 
    # getting residuals
    residuals = model.resid
    
    # Heteroscedasticity tests
    print("Heteroscedasticity Tests:")
    
    # Breusch-Pagan test
    bp_test = het_breuschpagan(residuals, model.model.exog)
    print(f"Breusch-Pagan Test: LM Statistic = {bp_test[0]}, p-value = {bp_test[1]}")
    
    # White test
    white_test = het_white(residuals, model.model.exog)
    print(f"White Test: LM Statistic = {white_test[0]}, p-value = {white_test[1]}")
    print("\n" + "="*50 + "\n")
    
    # Autocorrelation tests
    print("Autocorrelation Tests:")
    
    # Durbin-Watson test
    dw_test = durbin_watson(residuals)
    print(f"Durbin-Watson Test Statistic: {dw_test}")
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("Ljung-Box Test:")
    print(lb_test)
    print("\n" + "="*50 + "\n")
 # Function end

# Define the regression formula (Static Model) (model_1)

formula = 'GFE ~ LFE + EDP + weekend + peak_weekday'

# Fit the regression model
model = smf.ols(formula, data=df).fit()

# Print the summary of the regression
print(model.summary())

# Export static model results as an image
export_results_as_image(model, 'model_1.png')

# i am calling a function created above for assessig model diagnostics
regression_diagnostics(model)



# Create lagged variables (one-period lag)
df['LFE_lag1'] = df['LFE'].shift(1)  # One-period lag of LFE
df['EDP_lag1'] = df['EDP'].shift(1)  
df['GFE_lag1'] = df['GFE'].shift(1)  

# Drop the first row (since lagged values will have NaN for the first observation)
df = df.dropna()

# Plot ACF and PACF for GFE
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(df['GFE'], lags=20, ax=plt.gca(), title='Autocorrelation Function (ACF) for GFE')
plt.subplot(2, 1, 2)
plot_pacf(df['GFE'], lags=20, ax=plt.gca(), title='Partial Autocorrelation Function (PACF) for GFE')
plt.tight_layout()
plt.show()

# Finite Distributed lag model with dummy variables ARDL(1) (model_2)
# Define the regression formula
formula = 'GFE ~ LFE + EDP + weekend + peak_weekday + GFE_lag1 + LFE_lag1 + EDP_lag1'

# Fit the regression model
model = smf.ols(formula, data=df).fit()

# Print the summary of the regression
print(model.summary())

# Export static model results as an image
export_results_as_image(model, 'model_2.png')

# i am calling a function created above for assessig model diagnostics
regression_diagnostics(model)


# Model 3
# Create second-order lagged variables for ARDL(4) (model_3)

df['GFE_lag2'] = df['GFE'].shift(2)  # Second lag of GFE
df['LFE_lag2'] = df['LFE'].shift(2)  # Second lag of LFE
df['EDP_lag2'] = df['EDP'].shift(2)  # Second lag of EDP

# Create second-order lagged variables for ARDL(4) (model_3)
df['GFE_lag3'] = df['GFE'].shift(3)  # Second lag of GFE
df['LFE_lag3'] = df['LFE'].shift(3)  # Second lag of LFE
df['EDP_lag3'] = df['EDP'].shift(3)  # Second lag of EDP

# Create second-order lagged variables for ARDL(4) (model_3)
df['GFE_lag4'] = df['GFE'].shift(4)  # 4th lag of GFE
df['LFE_lag4'] = df['LFE'].shift(4)  
df['EDP_lag4'] = df['EDP'].shift(4)  

# Drop rows with NaN values (due to lagging)
df = df.dropna()

# Define the regression formula
formula = 'GFE ~ LFE + EDP + weekend + peak_weekday + GFE_lag1 + LFE_lag1 + \
           EDP_lag1 + GFE_lag2 + LFE_lag2 + EDP_lag2 + GFE_lag3 + LFE_lag3 + \
           EDP_lag3 + GFE_lag4 + LFE_lag4 + EDP_lag4'
           

# Fit the regression model
model = smf.ols(formula, data=df).fit()

# Print the summary of the regression
print(model.summary())

# Export static model results as an image
export_results_as_image(model, 'model_3.png')

#-----------Comparing ARDL (4, unconditional against AR(1) The Benchmark model)
#Fitting the benchmark AR(1) model

# Define the regression formula
formula = formula='GFE ~ GFE_lag1'
# Fit the regression model
model = smf.ols(formula, data=df).fit()

# Print the summary of the regression
print(model.summary())

# Export static model results as an image
export_results_as_image(model, 'AR(1) The Benchmark Model')
# i am calling a function created above for assessig model diagnostics
regression_diagnostics(model)

#Fitting the benchmark ARDL(4, unconditional) model

# Define the regression formula
formula = 'GFE ~ weekend + peak_weekday + GFE_lag1 + LFE_lag1 + \
           EDP_lag1 + GFE_lag2 + LFE_lag2 + EDP_lag2 + GFE_lag3 + LFE_lag3 + \
           EDP_lag3 + GFE_lag4 + LFE_lag4 + EDP_lag4'
# Fit the regression model
model = smf.ols(formula, data=df).fit()

# Print the summary of the regression
print(model.summary())

# Export static model results as an image
export_results_as_image(model, 'AR(4, unconditional) The Best performing model')

# i am calling a function created above for assessig model diagnostics
regression_diagnostics(model)

#-----------



# Splitting the data into training and testing sets
train_size = int(len(df) * 0.8)  # 80% for training, 20% for testing
train_data = df[:train_size].copy()  # i m using .copy() to avoid copy warning
test_data = df[train_size:].copy()   


df_time = pd.DataFrame()
df_time['time'] = df.index

# adding the 'time' column to train_data and test_data (only the corresponding rows)
train_data['time'] = df_time['time'][:train_size]
test_data['time'] = df_time['time'][train_size:]

# Defining function for model evaluation
# Function to evaluate model performance
def evaluate_model(actual, predicted, model_name):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    print(f'\nPerformance Metrics for {model_name}:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
# Fuction end...

# Model evaluation section starts here.
# Model 1: AR(1)
reg_1 = smf.ols(formula='GFE ~ GFE_lag1', data=train_data)
results_1 = reg_1.fit()
test_data['GFE_pred_ar1'] = results_1.predict(test_data)

# Model 2: Static Model
formula2 = 'GFE ~ LFE + EDP + weekend + peak_weekday'

# Fit the regression model
model2 = smf.ols(formula2, data=df).fit()
test_data['GFE_pred_static'] = model2.predict(test_data)

# I will not use static model in forecasting and in the performance evaluation of models

# Model 3:  ARDL(1)
# Define the regression formula
formula3 = 'GFE ~ LFE + EDP + weekend + peak_weekday + GFE_lag1 + LFE_lag1 + EDP_lag1'

# Fit the regression model
model3 = smf.ols(formula3, data=df).fit()
test_data['GFE_pred_ardl1'] = model3.predict(test_data)


# Model 4: ARDL 4 (Conditional)
formula4 = 'GFE ~ LFE + EDP + weekend + peak_weekday + GFE_lag1 + LFE_lag1 + \
           EDP_lag1 + GFE_lag2 + LFE_lag2 + EDP_lag2 + GFE_lag3 + LFE_lag3 + \
           EDP_lag3 + GFE_lag4 + LFE_lag4 + EDP_lag4'
           


model4 = smf.ols(formula4, data=df).fit()

test_data['GFE_pred_ardl4'] = model4.predict(test_data)

# Model 5: ARDL 4 (droping current values to make the forecast unconditional)
formula5 = 'GFE ~ weekend + peak_weekday + GFE_lag1 + LFE_lag1 + \
           EDP_lag1 + GFE_lag2 + LFE_lag2 + EDP_lag2 + GFE_lag3 + LFE_lag3 + \
           EDP_lag3 + GFE_lag4 + LFE_lag4 + EDP_lag4'
           


model5 = smf.ols(formula5, data=df).fit()

test_data['GFE_pred_ardl4_uncondional'] = model5.predict(test_data)


# Model 6: ARIMA(1,0,1)
arima_model = ARIMA(train_data['GFE'], order=(1, 0, 1))  # ARIMA(p=1, d=1, q=1)
arima_results = arima_model.fit()
test_data['GFE_pred_arima'] = arima_results.forecast(steps=len(test_data))


# Model 7: ARMA(4,1)
arma41_model = ARIMA(train_data['GFE'], order=(4, 0, 1))  # ARMA(p=4, q=1)
arma41_results = arma41_model.fit()
test_data['GFE_pred_arma41'] = arma41_results.forecast(steps=len(test_data))

# Model 8: ARMAX(4,1)
# Note: ARMAX is not directly supported in statsmodels, so we use ARIMA with exogenous variables
armax41_model = ARIMA(train_data['GFE'], exog=train_data[['LFE', 'EDP', 'weekend', 'peak_weekday']], order=(4, 0, 1))  # ARMAX(p=4, q=1)
armax41_results = armax41_model.fit()
test_data['GFE_pred_armax41'] = armax41_results.forecast(steps=len(test_data), exog=test_data[['LFE', 'EDP', 'weekend', 'peak_weekday']])

# Model 9: VAR (GFE, LFE, EDP) with lag 4
var_data = df[['GFE', 'LFE', 'EDP']]
optimal_lag = 4  # Set lag to 4
var_model_fitted = VAR(var_data[:train_size]).fit(maxlags=optimal_lag)
lag_order = var_model_fitted.k_ar
var_forecast = var_model_fitted.forecast(var_data.values[-lag_order:], steps=len(test_data))
test_data['GFE_pred_var'] = var_forecast[:, 0]  # Extract GFE forecasts


print(" \n\n\n Model performance evaluation for within the sample forecast \n\n\n")

# Evaluate all models
evaluate_model(test_data['GFE'], test_data['GFE_pred_ar1'], 'AR(1) Model')
evaluate_model(test_data['GFE'], test_data['GFE_pred_ardl1'], 'ARDL(1) Model')
evaluate_model(test_data['GFE'], test_data['GFE_pred_ardl4'], 'ARDL(4) Model') 
evaluate_model(test_data['GFE'], test_data['GFE_pred_ardl4_uncondional'], 'ARDL(4_uncondional) Model') 
evaluate_model(test_data['GFE'], test_data['GFE_pred_arima'], 'ARIMA(1,0,1) Model')
evaluate_model(test_data['GFE'], test_data['GFE_pred_arma41'], 'ARMA(4,1) Model')
evaluate_model(test_data['GFE'], test_data['GFE_pred_armax41'], 'ARMAX(4,1) Model')
evaluate_model(test_data['GFE'], test_data['GFE_pred_var'], 'VAR Model (Lag 4)')


print("\n In this model selection exercise ARDL(4, unconditional) performs well and can be used for causal effects, statistical inference and forecasting.")


print("\n\n\n It is the end for now!")











