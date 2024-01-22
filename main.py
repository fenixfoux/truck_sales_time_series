"""
inspired by https://www.kaggle.com/code/ddosad/time-series-arma-arima-sarima-concepts/notebook 
"""

import sys
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import calendar
import ploting_functions as pf

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

DATA_FILEPATH = 'files/Truck_sales.csv'

def preparing_data(data_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(data_filepath)
    # print(data)
    dates = pd.date_range(start='2003-01-01', freq='MS', periods=len(data))
    print(type(dates))

    # creating new columns with month and date
    data['Month'] = dates.month
    data['Month'] = data['Month'].apply(lambda x: calendar.month_abbr[x])  # spltting into english month names
    data['Year'] = dates.year

    # drop the original month-year column
    data.drop(['Month-Year'], axis=1, inplace=True)
    data.rename(columns={'Number_Trucks_Sold':'Truck-Sales'}, inplace=True)  # renaming Number_Trucks_Sold column

    # Only the necessary columns 'Month', 'Year', 'Truck-Sales' are selected,
    # and an index is set using the dates time series
    data = data[['Month', 'Year', 'Truck-Sales']]
    data.set_index(dates, inplace=True)
    # print(data.head())

    # plotting truck sales column to see if there is a trend and multiplicative seasonality
    pf.plot_truck_sales(data)
    # there are observed both, trend and multiplicative seasonality
    """
    Check for Stationarity?

    Dickey-Fuller Test - Dicky Fuller Test on the timeseries is run to check for stationarity of data.
    - Null Hypothesis H0 : Time Series is non-stationary.
    - Alternate Hypothesis Ha : Time Series is stationary.
    So Ideally if p-value < 0.05 then null hypothesis: TS is non-stationary is rejected else the TS is non-stationary is failed to be rejected.
    """
    from statsmodels.tsa.stattools import adfuller
    sales_ts = data['Truck-Sales']
    dftest = adfuller(sales_ts)  # python implementation of dicky fuller test
    print(dftest)
    print('DF test statistic is %3.3f' % dftest[0])
    print('DF test p-value is %1.4f' % dftest[1])
    # The p-value 0.99 is very large, and not smaller than 0.05 and thus is not a stationary series

    """
    Performing the decomposition of data if there is an existence of seasonality and split the data accordingly.
    Observe how number of Trucks sold vary on a month on month basis. A stacked plot for every year will give us a clear 
    pattern of any seasonality over the many years and those changes will be clearly reflected in the plots.
    """
    pf.bivariate_analyse(data)
    """
    # Inferences
    #     - The Truck sales have been increasing every year.
    #     - July/August are the peak months for sales
    #     - Variance & Mean values in the above 2 months are also higher than any of the other months.
    #     - 12 months seasonal cycle is present as mean of each month year on year starts with an increasing trend in 
            the beginning of the year and drops down towards the end of the year.
    """

    # Time Series Decomposition
    decomposition = sm.tsa.seasonal_decompose(sales_ts, model='multiplicative')
    pf.time_series_decomposition(decomposition)
    """
    Key observations:
        1) Trend: 12-months MA is a fairly straight line indicating a linear trend.
        2) Seasonality: seasonality of 12 months is clearly visible
        3) Irregular Remainder (random): The multiplicative model works as there are no patterns in the residuals
    """

    """
    How to Make a Time Series Stationary ? -> Differencing 'd'
        - Differencing 'd' is done on a non-stationary time series data one or more times to convert it into stationary.
        - (d=1) 1st order differencing is done where the difference between the current and previous (1 lag before) 
            series is taken and then checked for stationarity using the ADF(Augmented Dicky Fueller) test. If 
            differenced time series is stationary, we proceed with AR modeling. Else we do (d=2) 2nd order differencing,
            and this process repeats till we get a stationary time series
        - 1st order differencing equation is :  yt=yt−yt−1
        - 2nd order differencing equation is :  yt=(yt−yt−1)−(yt−1−yt−2)
          and so on…
        - The variance of a time series may also not be the same over time. To remove this kind of non-stationarity, we 
            can transform the data. If the variance is increasing over time, then a log transformation can stabilize 
            the variance.
    """
    # Show Non differenced full data Time series
    # pf.plot_truck_sales(sales_ts)

    print(f"="*99)
    pf.show_truck_sales(sales_ts)
    # Performing differencing ( d=1 ) as the data is non-stationary
    print(f"Performing differencing ( d=1 ) as the data is non-stationary\n")
    pf.make_timeseries_stationary(sales_ts, 1)
    # Try for d=2
    print(f"TPerforming differencing ( d=2 ) as the data is non-stationary\n")
    pf.make_timeseries_stationary(sales_ts, 2)

    # We observe seasonality even after differencing. Meaning the variance in the data seemss to be increasing.
    # This suggests a log transformation of the data
    print(f"as we observe seasonality even after differencing, it suggests to log transformation of the data\n")
    pf.log_transforming_data(sales_ts)

    # We observe trend and seasonality even after taking log of the observations.
    # Performing differencing (d=1) on the log transformed time series
    print(f"We observe trend and seasonality even after taking log of the observations.\n"
          f"Performing differencing (d=1) on the log transformed time series\n")
    pf.make_timeseries_stationary(pf.log_transforming_data(sales_ts), 1)

    print(f"Performing differencing (d=2) on the log transformed time series\n")
    pf.make_timeseries_stationary(pf.log_transforming_data(sales_ts), 2)



    prepared_data = data.copy()
    return prepared_data


































preparing_data(DATA_FILEPATH)

























