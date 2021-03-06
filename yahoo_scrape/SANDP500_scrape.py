# SOURCE https://towardsdatascience.com/free-stock-data-for-python-using-yahoo-finance-api-9dafd96cad2e
# SOURCE https://medium.com/analytics-vidhya/scraping-s-p-500-index-and-extracting-stock-market-data-from-yahoo-finance-api-72218eeed1be
# TODO https://github.com/rjagait/Stock-Market-Prediction
# TODO https://www.learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
# TODO https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
# TODO https://pythonrepo.com/repo/VivekPa-NeuralNetworkStocks-python-deep-learning
# TODO https://pypi.org/project/Backtesting/
# TODO https://towardsdatascience.com/backtest-your-trading-strategy-with-only-3-lines-of-python-3859b4a4ab44
# TODO https://www.datacamp.com/community/tutorials/finance-python-trading
# TODO https://algotrading101.com/learn/backtrader-for-backtesting/
# TODO https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/
# TODO https://codingandfun.com/backtesting-with-python/
# TODO https://kernc.github.io/backtesting.py/
# TODO https://www.youtube.com/watch?v=xfzGZB4HhEE
# TODO analyse news and clean up with https://hub.gke2.mybinder.org/user/residentmario-missingno-5wr46gm8/notebooks/QuickStart.ipynb
# TODO setup https://christoolivier.com/what-is-up-with-__init__-py-and-__main__-py/ for each module
# TODO https://support.apple.com/en-lb/guide/mac-help/mchlp1066/mac
# TODO https://www.codementor.io/@mauhcs/train-neural-networks-faster-with-google-s-tpu-from-your-laptop-19e2gr17nv
# TODO https://github.com/alvarobartt/trendet/
# TODO https://investpy.readthedocs.io/_info/related_projects.html
# TODO an option for individuals https://help.yahoo.com/kb/finance-for-web/download-historical-data-yahoo-finance-sln2311.html
# https://github.com/mementum/backtrader

# jupyternotebook and google colab similar format.
# TODO https://www.datasciencecentral.com/profiles/blogs/all-about-using-jupyter-notebooks-and-google-colab
# TODO TODO TODO FIRST FIRST FIRST https://developers.google.com/machine-learning/crash-course/

import bs4 as bs
import pickle # remove pickle and sub in csv
import requests
import datetime as dt
import os
import pandas_datareader.data as pdr

import numpy
import lxml
import getpass


def main():
    save_sp500_tickers()
    get_data_from_yahoo()
    # TODO fix error handling for sp500tickers variable so it doesn't show as red error
    sp500tickers.pickle = open("csvfile.csv", " wb")


def save_sp500_tickers():
    """A function to save sp500 data as pickle"""
    # TODO modify to change to csv
     #error handling, 200 = good, 404 bad etc
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f: # TODO remove pickle swap csv
        pickle.dump(tickers, f)
    print(resp)
    return tickers


def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(1950, 1, 1)
    end = dt.datetime.now()

    for ticker in tickers:
        if not os.path.exists(CSV_SAVE_PATH + 'stock_dfs/{}.csv'.format(ticker)):
            df = pdr.DataReader(ticker.replace('.', '-'), 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv(CSV_SAVE_PATH + 'stock_dfs/{}.csv'.format(ticker)) # TODO save to /Data
        else:
            print('Already have {}'.format(ticker))


def error_handling():
    """a way to guage progress or issues"""
    # should get <Response [200]>
    # immediately followed by stocks_dfs folder being made and populated.
    

def yahoo_logfile():
    """create a csv log file of date, time, run number, stocks updated, elapsed time, port number etc."""
    
    
"""
def five_min_gap(): # TODO P7PF-3
    tickers = [  # list of 1000+ stock tickers]
        for ticker in tickers:
    data = yf.download(tickers=ticker, period='30m', interval='1m')
    output.append({"Ticker": ticker, "Price": data.tail(1)['Open'].values[0], "Volume":
        data.tail(1)['Volume'].values[0]})
"""

if __name__ == "__main__":
    user_name = getpass.getuser() # TODO fix this
    CSV_SAVE_PATH = '~/Users/wadewilson/Library/Mobile Documents/com~apple~CloudDocs/Data/Financial/yahoo_sp500/'
    DATA_PATH = f'/Users/wadewilson/Library/Mobile Documents/com~apple~CloudDocs/Data/Financial' # TODO FIXME b4 running
    # DATA_PATH = f'/Users/wadewilson/Data/yahoo_finance_Sp500_scrape/'
    # DATA_PATH = f'/Users/{}/Data/yahoo_finance_Sp500_scrape/'.format(user_name) #FIXME P7PF-2
    main()

    