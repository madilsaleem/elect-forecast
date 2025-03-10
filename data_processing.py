# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:48:07 2025

@author: Muhammad Adil Saleem
Purpose: Data Processing for modelling

Note: "select" code to run step by step or "select all code" to run at once
"""
# Creating a minutes minutes variables
import pandas as pd
import numpy as np
import matplotlib as mp
import os

# figure out current working directory,
# Change the wd to the folder of technical challenge "Extract first if zipped"
# Change the wd to the folder of technical challenge "Extract first if zipped"
cwd = os.getcwd()
print("Current working directory:", cwd)
#if needed, change it to desired wd
os.chdir(r'C:\Users\user\Enoda-forecasting')

#import excel file and read a worksheet 
excel_file = 'Spain Electricity Data.xlsx'  
sheet_name = 'Generation'  
sheet_name1= 'Dayahead_prices' # fix from here, read all data into dfs first then manipulate the rest.
sheet_name2= 'Load'
df1 = pd.read_excel(os.path.join(cwd, excel_file), sheet_name=sheet_name)
df2 = pd.read_excel(os.path.join(cwd, excel_file), sheet_name=sheet_name1, usecols=['Day-ahead (EUR/MWh)'])
df3 = pd.read_excel(os.path.join(cwd, excel_file), sheet_name=sheet_name2,
                   usecols=['Day ahead total load forecast (MW)',
                            'Actual total load (MW)'])


# ignoring n/e and empty cells and summing generation units, create new df_sum
df1_sum = df1.replace(['n/e', ''], np.nan)

#    convert all columns to numeric values (suppressing errors)
df1_sum = df1_sum.apply(pd.to_numeric, errors='coerce')


# summing electricity generation from each source:
df1['Total_actual_gen'] = df1_sum[['Biomass - Actual Aggregated [MW]',
                             'Energy storage - Actual Aggregated [MW]',
                             'Fossil Brown coal/Lignite - Actual Aggregated [MW]',
                             'Fossil Coal-derived gas - Actual Aggregated [MW]',
                             'Fossil Gas - Actual Aggregated [MW]',
                             'Fossil Hard coal - Actual Aggregated [MW]',
                             'Fossil Oil - Actual Aggregated [MW]',
                             'Fossil Oil shale - Actual Aggregated [MW]',
                             'Fossil Peat - Actual Aggregated [MW]',
                             'Geothermal - Actual Aggregated [MW]',
                             'Hydro Pumped Storage - Actual Aggregated [MW]',
                             'Hydro Pumped Storage - Actual Consumption [MW]',
                             'Hydro Run-of-river and poundage - Actual Aggregated [MW]',
                             'Hydro Water Reservoir - Actual Aggregated [MW]',
                             'Marine - Actual Aggregated [MW]',
                             'Nuclear - Actual Aggregated [MW]',
                             'Other - Actual Aggregated [MW]',
                             'Other renewable - Actual Aggregated [MW]',
                             'Solar - Actual Aggregated [MW]',
                             'Waste - Actual Aggregated [MW]',
                             'Wind Offshore - Actual Aggregated [MW]',
                             'Wind Onshore - Actual Aggregated [MW]'
                             ]].sum(axis=1)
print(df1['Total_actual_gen'].head())

# differencing Total generation minus Total Day ahead generation forecast

df1['Generation forecast error']= df1['Total_actual_gen'].sub(df1['Scheduled Generation [MW] (D) - BZN|ES'], fill_value=0)
 
print(df1[['Total_actual_gen', 'Scheduled Generation [MW] (D) - BZN|ES',
         'Generation forecast error' ]].head())

# differencing Total Load minus Total Day ahead Load forecast
df3_sum = df3.replace(['n/e', ''], np.nan) 
df3_sum = df3_sum.apply(pd.to_numeric, errors='coerce') 
df3['Load forecast error']= df3_sum['Actual total load (MW)'].sub(df3_sum['Day ahead total load forecast (MW)'], fill_value=0)
print(df3['Load forecast error'].head())

# Combining varaible of interest from DataFrame df1, df2, df3 into cobmined_df

combined_df= pd.DataFrame()
combined_df['Generation forecast error']= df1['Generation forecast error']
combined_df['Day-ahead (EUR/MWh)']= df2['Day-ahead (EUR/MWh)']
combined_df['Load forecast error']= df3['Load forecast error']
 
# Now we will find Nan, missing and 0 values in Generation forecast error
missing_Nan_GFE = combined_df[combined_df['Generation forecast error'].isna()
                              | (combined_df['Generation forecast error'] == '')
                              |(combined_df['Generation forecast error'] == 0)].index

print(missing_Nan_GFE)

# we will only fill values at 8648, 8649, 8650, 8651 due missing data,
# Other indices are zero because the forecast error was actually zero, I figure out by inspecting the data 
# now we will fill the values by 

# set subsample from 8600 to 8651
subset = combined_df.loc[8600:8651]

# Calculating the four-point moving average for the missing indices
for i in [8648, 8649, 8650, 8651]:
    # using the previous 4 values to calculate the moving average
    moving_avg = subset.loc[i-4:i-1, 'Generation forecast error'].mean()
    # filling the missing value with the moving average
    subset.at[i, 'Generation forecast error'] = moving_avg

# now we will update the original DataFrame with the filled values
combined_df.loc[8600:8651] = subset

# Print the filled values at the specified indices, where the values were missing
print(combined_df.loc[[8648, 8649, 8650, 8651]])

# now let's finding missing, Nan, zero values in 'Load Forecast Error'
# Now we will find Nan, missing and 0 values in Generation forecast error
missing_Nan_LFE = combined_df[combined_df['Load forecast error'].isna()
                              | (combined_df['Load forecast error'] == '')
                              |(combined_df['Load forecast error'] == 0)].index

print(missing_Nan_LFE)

# All are genuinely zero i.e load forecast = actual load except,
# those four points 8648, 8649, 8650, 8651, It seems like there was some ..
# problem in spanish grid on 31-03-2024 b/w 02:00 to 03:00, but I have no time to figure out now.

# set subsample again from 8600 to 8651
subset = combined_df.loc[8600:8651]

# Calculating the four-point moving average for the missing indices
for i in [8648, 8649, 8650, 8651]:
    # Use the previous 4 values to calculate the moving average
    moving_avg = subset.loc[i-4:i-1, 'Load forecast error'].mean()
    # Fill the missing value with the moving average
    subset.at[i, 'Load forecast error'] = moving_avg

# Update the original DataFrame with the filled values
combined_df.loc[8600:8651] = subset

# Print the filled values at the specified indices
print(combined_df.loc[[8648, 8649, 8650, 8651]])

# now we are good to go with the missing values, they are all filled

"""Now turn to aggregation/disaggreagtion to 30 mins interval.
 
we will aggregate Generation load error and load forecast error from 15 mins to 30 mins.
It simple for aggregation, we just need to add two 15 mins interval to form 30 minutes
interval. This will not distort the nature of the data. (not agreed?, let's discuss!')
"""
# now adding two successive elements of each variable
combined_df['GFE'] = combined_df['Generation forecast error'].rolling(window=2).sum()
combined_df['LFE'] = combined_df['Load forecast error'].rolling(window=2).sum()

print(combined_df[['GFE', 'LFE']].head())

# Day-ahead prices are in hourly. I will split in 30 mins interval by diving into 2.
# Since our obective is to convert it into 30 minutes, however electricy is bidded at 1 hour interval.
#Therefore, we will have same price for two successive periods.

# EDP = Electricity Day-ahead Prices in 30 minutes interval
combined_df['EDP'] = combined_df['Day-ahead (EUR/MWh)'].repeat(2).reset_index(drop=True)
print(combined_df['EDP'].head())


# Define start and end dates
start_date = '2024-01-01'
end_date = '2025-02-21'

# Create a date range with 30-minute intervals
time_range = pd.date_range(start=start_date, end=end_date, freq='30T')

# Truncate combined_df to match the length of time_range
cdf = combined_df.iloc[:len(time_range)]

# Add the time range to the DataFrame
cdf['time'] = time_range

print(cdf.head())

# creating a dummy variable for weekends (Saturday and Sunday)
# 1 =Weekend, 0 = Workday
cdf['weekend'] = cdf['time'].dt.dayofweek.isin([5, 6]).astype(int)

#  create a dummy variable for peak hours (10:00 to 14:00 and 18:00 to 22:00)
cdf['peak'] = (
    (cdf['time'].dt.time >= pd.to_datetime('10:00').time()) & 
    (cdf['time'].dt.time <= pd.to_datetime('14:00').time()) |
    (cdf['time'].dt.time >= pd.to_datetime('18:00').time()) & 
    (cdf['time'].dt.time <= pd.to_datetime('22:00').time())
).astype(int)

# create a dummy variable for peak hours only on weekdays
cdf['peak_weekday'] = ((cdf['weekend'] == 0) & (cdf['peak'] == 1)).astype(int)

# deleting column which is not needed...
cdf.drop(columns=['peak'], inplace=True)

# adding value of day ahead energy prices i.e 77.8 Euro/MWH for 21-02-2025 00:00 to complete the data
cdf.at[20016, 'EDP'] =77.8

# Saving the final variables in CSV file.
final_df = cdf[['time','GFE', 'LFE','EDP','weekend','peak_weekday' ]]
final_df.to_csv('data for modelling.csv', index=False)

print('\n\n The end for data processing exercise')