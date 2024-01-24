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

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import ploting_functions as pf

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

DATA_FILEPATH = 'files/Truck_sales.csv'
SHOW_PLOT = False


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
    data.rename(columns={'Number_Trucks_Sold': 'Truck-Sales'}, inplace=True)  # renaming Number_Trucks_Sold column

    # Only the necessary columns 'Month', 'Year', 'Truck-Sales' are selected,
    # and an index is set using the dates time series
    data = data[['Month', 'Year', 'Truck-Sales']]
    data.set_index(dates, inplace=True)
    # print(data.head())

    # plotting truck sales column to see if there is a trend and multiplicative seasonality
    if SHOW_PLOT: pf.plot_truck_sales(data)
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
    if SHOW_PLOT: pf.bivariate_analyse(data)
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
    if SHOW_PLOT: pf.time_series_decomposition(decomposition)
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

    print(f"=" * 99)
    if SHOW_PLOT:
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


"""
   ============================== Auto Regressive(AR) Models ==============================
    - Autoregression means regression of a variable on itself which means Autoregressive models use previous time period
      values to predict the current time period values.
    - One of the fundamental assumptions of an AR model is that the time series is assumed to be a stationary process.
    - An AR(p) model (Auto-Regressive model of order p) can be written as:
        Yt = φ1Yt−1+φ2Yt−2+……+φpYt−p+εt
     
    -  εt
        is an error term which is an independent and identically distributed random variable (or in other words, a white 
        noise) with the parameters mean = 0 and standard deviation = σ - The φ are regression coefficients multiplied by 
        lagged time series variable, which captures the effect of the input variable on the output, provided intermediate 
        values do not change.
      
    | Choose the order 'p' of AR model
    - We look at the Partial Autocorrelations of a stationary Time Series to understand the order of Auto-Regressive 
        models.
    - For an AR model, 2 ways to identify order of 'p':
    1) PACF Approach : the PACF method where the (Partial Auto Correlation Function) values cut off and become zero 
        after a certain lag. PACF vanishes if there is no regression coefficient that far back. The cut-off value where 
        this happens can be taken as the order of AR as ‘p’. This can be seen from a PACF plot.
    - If the 2nd PACF vanishes (cut off in PACF) then the 2nd coefficient is not considered and thus ‘p’ is 1. 
    - If the 3rd PACF vanishes (cuts off in PACF) then the 3rd coefficient is not considered and thus ‘p’ is 2 and so on…
    - Partial Autocorrelation of order 2 = Partial autocorrelation of lag 2 = Correlation between Xt and 
        Xt−2 holding Xt−1 fixed.
    2) Lowest AIC Approach : the lowest Akaike Information Criteria (AIC) value compared among 
        different orders of ‘p’ is considered.
"""

# Using the 2nd method(Lowest AIC) to compare different orders of 'p'
# Define the p parameter to take any value between 0 and 2
p = range(1, 4)

"""
============================== Moving Average(MA) Models  ==============================
    - Moving average model considers past residual values to predict the current time period values,
        These past residuals are past prediction errors.
    - For a MA model, the residual or error component is modeled
    - The moving average model MA(q) of  order q can be represented as: yt=εt+θ1εt−1+……+θqεt−q
    - Where  yt time series variable, θ are numeric coefficients multiplied to lagged residuals and ε is the residual 
        term considered as a purely random process with mean 0, variance  σ2 and   Cov(εt−1,εt−q) = 0.
        
    | Choose the order 'q' of MA model
    - We look at the Autocorrelations of a stationary Time Series to understand the order of Moving Average models.
    - For a MA model
    1) ACF Approach : ACF (Autocorrelation Function) values cut off at a certain lag. ACF vanishes, and there are no 
        coefficients that far back; thus, the cut-off value where this happens is taken as the order of MA as ‘q’. 
        This can be seen from the ACF plot.
    2) Lowest AIC Approach : the lowest Akaike Information Criteria (AIC) value compared among different 
        orders of ‘q’ is considered.
"""

# Lowest AIC Method#
# Using the 2nd method(Lowest AIC) to compare different orders of 'q'
# Define the q parameter to take any value between 0 and 2
q = range(1, 4)
"""
    WHY?
    
    **ACF(0)=1**
    **ACF(1)=PACF(1)**
    Parameter (p, d, q) estimation matrix for estimating parameters towards building AR / ARMA / ARIMA / SARIMA models
"""
d = range(0, 2)

# Generate all different combinations of p with d=0 and q=0 triplets for AR model building
pdq_ar = list(itertools.product(p, range(1), range(1)))
print(f"pdq_ar: {pdq_ar}")

# Generate all different combinations of p,q with d=0 triplets for ARMA model building
pdq_arma = list(itertools.product(p, range(1), q))
print(f"pdq_arma: {pdq_arma}")

# Generate all different combinations of p, d and q triplets for ARIMA model building
pdq = list(itertools.product(p, d, q))
print(f"pdq: {pdq}")

# Generate all different combinations of seasonal P,D,Q triplets for SARIMA model building
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print(f"seasonal_pdq: {seasonal_pdq}")

data = preparing_data(DATA_FILEPATH)

# Separate data into train and test
data['date'] = data.index
train = data[data.index < '2013-01-01']
test = data[data.index >= '2013-01-01']

print(train.head())
print(test.head())
print("=" * 99)

dftest = adfuller(train['Truck-Sales'])  # Stationarity check on train time series data
print(dftest)
print('DF test statistic is %3.3f' % dftest[0])
print('DF test p-value is %1.4f' % dftest[1])

train_sales_ts_log = np.log10(train['Truck-Sales'])
# Log transformation on the training data to make the time series stationary as we did with full data


best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

# ============================ AR Model : Autoregressive ============================

# - Use previous time period values to predict the current time period values AR Model building to
#   estimate best 'p' ( Lowest AIC Approach )


# Creating an empty Dataframe with column names only
AR_AIC = pd.DataFrame(columns=['param', 'AIC'])

for param in pdq_ar:
    ARIMA_model = ARIMA(train_sales_ts_log, order=param).fit()
    print('ARIMA{} - AIC:{}'.format(param, ARIMA_model.aic))
    row = {'param': param, 'AIC': ARIMA_model.aic}
    AR_AIC = pd.concat([AR_AIC, pd.DataFrame([row])], ignore_index=True)

# Building AR model with best 'p' parameter
best_model = ARIMA(train_sales_ts_log, order=(2, 0, 0))  # p=2 with lowest AIC
best_results = best_model.fit()
print(best_results.summary().tables[0])
print(best_results.summary().tables[1])


# Calculating RMSE for best AR model
pred_dynamic = best_results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=True, full_results=True)
pred99 = best_results.get_forecast(steps=len(test), alpha=0.1) # forecasting values

# Extract the predicted and true values of our time series
sales_ts_forecasted = pred_dynamic.predicted_mean
testCopy1 = test.copy()
testCopy1['sales_ts_forecasted'] = np.power(10, pred99.predicted_mean)

# Compute the root-mean-square error
mse = ((testCopy1['Truck-Sales'] - testCopy1['sales_ts_forecasted']) ** 2).mean()
rmse = np.sqrt(mse)
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 3)))

axis = train['Truck-Sales'].plot(label='Train Sales', figsize=(15, 5))
testCopy1['Truck-Sales'].plot(ax=axis, label='Test Sales', alpha=0.7)
testCopy1['sales_ts_forecasted'].plot(ax=axis, label='Forecasted Sales', alpha=0.7)
axis.set_xlabel('Years')
axis.set_ylabel('Truck Sales')
plt.legend(loc='best')
plt.show()
plt.close()

resultsDf = pd.DataFrame({'RMSE': rmse}
                           ,index=['Best AR Model : ARIMA(2,0,0)'])

print(f"result of R model: {resultsDf}")

# ============================ ARMA Model ============================
































