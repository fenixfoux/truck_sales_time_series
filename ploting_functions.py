# plotting_functions.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_truck_sales(data):
    sales_ts = data['Truck-Sales']
    sns.set(style="white", rc={'figure.figsize': (12, 6)})
    plt.figure(figsize=(15, 4))
    plt.plot(sales_ts)
    plt.xlabel('Years')
    plt.ylabel('Truck Sales')
    plt.show()


def plot_monthly_sales(monthly_sales_data):
    sns.set(style="white", rc={'figure.figsize': (12, 6)})
    ax = monthly_sales_data.plot()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_facecolor('white')
    plt.show()


def bivariate_analyse(data):
    monthly_sales_data = pd.pivot_table(data, values="Truck-Sales", columns="Year", index="Month")
    # Reindexing
    monthly_sales_data = monthly_sales_data.reindex(
        index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    sns.set(style="white", rc={'figure.figsize': (12, 6)})
    ax = monthly_sales_data.plot()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_facecolor('white')
    plt.show()


    sns.set(style="white", rc={'figure.figsize': (12, 6)})
    monthly_sales_data.boxplot()
    plt.show()

    yearly_sales_data = pd.pivot_table(data, values="Truck-Sales", columns="Month", index="Year")
    yearly_sales_data = yearly_sales_data[
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

    sns.set(style="white", rc={'figure.figsize': (12, 6)})
    ax = yearly_sales_data.plot()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_facecolor('white')
    plt.show()

    sns.set(style="white", rc={'figure.figsize': (12, 6)})
    yearly_sales_data.boxplot()
    plt.show()


def time_series_decomposition(data):
    fig = data.plot()
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.suptitle('Decomposition of multiplicative time series')
    plt.show()


def show_truck_sales(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.xlabel('Years')
    plt.ylabel('Truck Sales')
    plt.show()


def make_timeseries_stationary(data, differencing: int):
    plt.figure(figsize=(12, 6))
    plt.plot(data.diff(periods=differencing))
    plt.xlabel('Years')
    plt.ylabel('Truck Sales')
    plt.show()


def log_transforming_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(np.log10(data))
    plt.xlabel('Years')
    plt.ylabel('Log (Truck Sales)')
    plt.show()
    return np.log10(data)







