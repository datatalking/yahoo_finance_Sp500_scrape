import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns



def main():
    DATA_PATH = '/Users/vanessawilson/Data/yahoo_finance_Sp500_scrape/stock_dfs/'
    SAMPLE_DATA_PATH = '/Users/vanessawilson/Data/yahoo_finance_Sp500_scrape/stock_dfs/'
    sphist = ''
    sphist = pd.read_csv(SAMPLE_DATA_PATH + 'sphist.csv')
    data_accessing(SAMPLE_DATA_PATH)
    data_features_pandsframe(sphist)
    data_newness_indicators(sphist)
    data_before_51_nan(sphist)
    data_train_and_test_split(sphist)
    linear_correlation(sphist)
    data_big_mse(sphist, train, test)
    data_random_forest(X_train, y_train, X_test, y_test)
    




def data_accessing(SAMPLE_DATA_PATH):
    """
    The dataset contains historical data on the price of the S&P500 Index. The columns are:
    
    Date: The date of the record.
    Open: The opening price of the day (when trading starts).
    High: The highest trade price during the day.
    Low: The lowest trade price during the day.
    Close: The closing price for the day (when trading is finished).
    Volume: The number of shares traded.
    Adj Close: The daily closing price, adjusted retroactively to include any corporate actions.
    """
    sphist = pd.read_csv(SAMPLE_DATA_PATH + 'sphist.csv')
    # sphist = pd.read_csv('../Data/yahoo_finance/Sp500_scrape/sphist.csv')
    print(sphist.describe())
    print("\ndf shape: ", sphist.shape)
    sphist.head()



def  data_features_pandsframe(sphist):
    """Convert 'Date' column to Pandas date type"""
    sphist['Date'] = pd.to_datetime(sphist['Date'])
    
    # Sort df by that column
    sphist.sort_values(by=['Date'], inplace=True)
    sphist.head()


def data_newness_indicators(sphist):
    """
    Given the nature of the stock market, in order to prevent injecting future knowledge into the model,
    let's create indicators based on the past.
    1) Average price for the last 5 days
    2) Average price for the last 365 days
    3) Ratio between the average price for the past 5 days, and the average price for the past 365 days.
    4) Standard deviation of the price for the last 5 days
    5) Standard deviation of the price for the last 365 days
    6) Ratio between the standard deviation for the past 5 days, and the standard deviation for the past 365 days.
    7) The average volume over the past five days.
    8) The average volume over the past year.
    9) The ratio between the average volume for the past five days, and the average volume for the past year.
    10) The ratio between the lowest price in the past year and the current price.
    :return:
    """
    # Add new indicators to each observation:
    # 1
    sphist['avg_price_5'] = sphist['Close'].rolling(5).mean()
    sphist['avg_price_5'] = sphist['avg_price_5'].shift()  # Avoid using current day's price by reindexing

    # 2
    sphist['avg_price_365'] = sphist['Close'].rolling(365).mean()
    sphist['avg_price_365'] = sphist['avg_price_365'].shift()  # Avoid using current day's price by reindexing

    # 3
    sphist['avg_price_5_365'] = sphist['avg_price_5'] / sphist['avg_price_365']

    # 4
    sphist['std_price_5'] = sphist['Close'].rolling(5).std()
    sphist['std_price_5'] = sphist['std_price_5'].shift()  # Avoid using current day's price by reindexing

    # 5
    sphist['std_price_365'] = sphist['Close'].rolling(365).std()
    sphist['std_price_365'] = sphist['std_price_365'].shift()  # Avoid using current day's price by reindexing

    # 6
    sphist['std_price_5_365'] = sphist['std_price_5'] / sphist['std_price_365']

    # 7
    sphist['avg_volume_5'] = sphist['Volume'].rolling(5).mean()
    sphist['avg_volume_5'] = sphist['avg_volume_5'].shift()  # Avoid using current day's price by reindexing

    # 8
    sphist['avg_volume_365'] = sphist['Volume'].rolling(365).mean()
    sphist['avg_volume_365'] = sphist['avg_volume_365'].shift()  # Avoid using current day's price by reindexing

    # 9
    sphist['avg_volume_5_365'] = sphist['avg_volume_5'] / sphist['avg_volume_365']

    # 10
    min_last_year = sphist['Close'].rolling(365).min()
    sphist['last_min_current_ratio'] = min_last_year / sphist['Close']
    sphist['last_min_current_ratio'] = sphist['last_min_current_ratio'].shift()


def data_before_51_nan(sphist):
    """
    Remove any rows from the DataFrame that fall before 1951-01-03
    Remove any rows with NaN values
    :return:
    """
    print("# of observations before: ", sphist.shape[0])
    print("NaN values before: \n\n", sphist.isnull().sum())
    
    sphist = sphist[sphist['Date'] > datetime(year=1951, month=1, day=2)]
    sphist.dropna(axis=0, inplace=True)
    
    print("\n# of observations after: ", sphist.shape[0])
    print("NaN values after: \n\n", sphist.isnull().sum())
    
    

def data_train_and_test_split(sphist):
    """
    Training set: Observations up to 2013-01-01
    Test set: Observations after 2013-01-01
    :return:
    """
    train = sphist[sphist["Date"] < datetime(year=2013, month=1, day=1)]
    test = sphist[sphist["Date"] >= datetime(year=2013, month=1, day=1)]

    print("Train: ", train.shape)
    print("Test: ", test.shape)
 
    return(train, test)
    

def linear_correlation(sphist):
    """
    # Sorted correlations with target column 'Close'

    :return:
    """
    sorted_corrs = sphist.corr()['Close'].sort_values()

    print(sorted_corrs)
    fig, ax = plt.subplots(figsize=(15,10))
    sns.heatmap(sphist[sorted_corrs.index].corr())
    return(sphist)


def data_big_mse(sphist, train, test):
    """
    
    :return:
    """
    features = ['avg_price_5', 'avg_price_365', 'avg_price_5_365', 'std_price_5',
                'std_price_365', 'std_price_5_365', 'avg_volume_5', 'avg_volume_365',
                'avg_volume_5_365', 'last_min_current_ratio']

    X_train = train[features]
    y_train = train['Close']

    X_test = test[features]
    y_test = test['Close']

    # Train
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict
    closing_price_pred_lr = lr.predict(X_test)

    # --------------------------------------------------
    # Performance metrics
    # --------------------------------------------------

    # Calculate MSE
    mse_lr = mean_squared_error(y_test, closing_price_pred_lr)

    # Calculate the absolute errors and MAPE
    errors_lr = abs(closing_price_pred_lr - y_test)
    mape_lr = 100 * (errors_lr / y_test)

    # MAE
    mae_lr = round(np.mean(errors_lr), 2)

    # Accuracy
    accuracy_lr = 100 - np.mean(mape_lr)

    print("-----------------\nLinear regression\n-----------------")
    print("MSE: ", mse_lr)
    print("MAE: ", mae_lr, "degrees")
    print('Accuracy:', round(accuracy_lr, 2), '%.')
    return(X_train, y_train, X_test, y_test)
    

def data_random_forest(X_train, y_train, X_test, y_test):
    """
    
    :return:
    """
    rf = RandomForestRegressor(n_estimators=150, random_state=1, min_samples_leaf=2)

    # Train
    rf.fit(X_train, y_train)

    # Predict
    closing_price_pred_rf = rf.predict(X_test)

    # --------------------------------------------------
    # Performance metrics
    # --------------------------------------------------

    # Calculate the absolute errors and MAPE
    errors_rf = abs(closing_price_pred_rf - y_test)
    mape_rf = 100 * (errors_rf / y_test)

    # MAE
    mae_rf = round(np.mean(errors_rf), 2)

    # Accuracy
    accuracy_rf = 100 - np.mean(mape_rf)

    print("-----------------\nRandom Forest\n-----------------")
    print("MAE: ", mae_rf, "degrees")
    print('Accuracy:', round(accuracy_rf, 2), '%.')


if __name__ == "__main__":
    main()
