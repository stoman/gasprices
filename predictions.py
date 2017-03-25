from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

from database import Database

def split_seasonal(history):
    """
    This function splits a time series of gas prices into a trend, a weekly
    pattern, and a residual. The three returned time series sum up to the
    input time series.

    Keyword arguments:
    history -- the time series to split up

    Return value:
    a tuple of three time series: trend, weekly, and residual
    """
    #compute trend
    length_week = 7*24
    trend = history.rolling(window=length_week, center=True, min_periods=1).mean()
    
    #compute the weekly changes
    weekly = pd.Series()
    for i in range(length_week):
        weekly = weekly.append((history - trend)[i::length_week].rolling(window=9, center=True, min_periods=1).mean())
    weekly.sort_index(inplace=True)
    
    #weekly = [(history - trend)[-length_week+i::-length_week].mean() for i in range(length_week)]
    #seasonal_uncut = np.tile(weekly, 1 + len(history) // length_week)
    #seasonal = seasonal_uncut[len(seasonal_uncut) - len(history):]
    #seasonal_series = pd.Series(seasonal, index=history.index)
    
    #compute residual
    res = history - trend - weekly
    
    return trend, weekly, res

def test_stationary(ts):
    """
    Print the results of the Dickey-Fuller test
    
    Named Arguments:
    ts -- the time series to analyze
    """
    result = adfuller(ts)
    print("Dickey-Fuller Test:")
    print("Test Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    for key, value in result[4].items():
        print("Critical Value for %s: %f" % (key, value))

def predict(history, hours=4*7*24, trend=None, weekly=None, res=None):
    """
    This function splits a time series of gas prices into a trend, a weekly
    pattern, and a residual. The three returned time series sum up to the
    input time series.

    Keyword arguments:
    history -- the time series to split up

    Return value:
    a tuple of three time series: trend, weekly, and residual
    """
    if trend is None or weekly is None or res is None:
        trend, weekly, res = split_seasonal(history)
    
    trend_shift = (trend - trend.shift(1)).fillna(0.)
    test_stationary(trend_shift)
    
    trend_model = ARIMA(trend_shift, order=(2, 1, 2))
    trend_results = trend_model.fit(disp=-1)
    trend_res = trend_results.predict(trend.index.max(), trend.index.max() + timedelta(hours=hours))
    trend_pred = trend_res.cumsum() + trend[-1]
    
    res_pred = pd.Series(index=trend_pred.index).fillna(0.)
    
    weekly_pred = pd.Series(index=trend_pred.index).fillna(0.)
    length_week = 7*24
    for i in range(length_week):
        weekly_pred[i::length_week] = weekly[-length_week + i]

    return trend_pred, weekly_pred, res_pred

def plot_split_seasonal(history, predictions=False):
    """
    This function plots an overview over the seasonal split of a history of gas
    prices as computed by `split_seasonal`.
    
    Keyword arguments:
    history -- the time series to split up and visualize
    predictions -- whether to plot predictions too (default False)
    """
    #split price history
    trend, weekly, res = split_seasonal(history)

    #predict price history
    if predictions:
        trend_pred, weekly_pred, res_pred = predict(history, trend=trend, weekly=weekly, res=res)
        history_pred = trend_pred + weekly_pred + res_pred

    #plot given time series
    palette = sns.color_palette(n_colors=6)
    ax = plt.subplot(3, 1, 1) 
    plt.plot(history, label="Price History", color=palette[1])
    plt.plot(trend, label="Trend", color=palette[2])
    if predictions:
        ax.axvline(history.index.max())
        plt.plot(history_pred, linestyle="dashed", color=palette[1])
        plt.plot(trend_pred, linestyle="dashed", color=palette[2])
    ax.legend(loc=1)
    
    ax = plt.subplot(3, 1, 2)
    plt.plot(history - trend, label="Price History without Trend", color=palette[3])
    plt.plot(weekly, label="Weekly Pattern", color=palette[4])
    if predictions:
        ax.axvline(history.index.max())
        plt.plot(history_pred - trend_pred, linestyle="dashed", color=palette[3])
        plt.plot(weekly_pred, linestyle="dashed", color=palette[4])
    ax.set_ylim([-100, 100])
    ax.legend(loc=1)

    ax = plt.subplot(3, 1, 3)
    plt.plot(res, label="Residual", color=palette[5])
    if predictions:
        ax.axvline(history.index.max())
        plt.plot(res_pred, linestyle="dashed", color=palette[5])
    ax.set_ylim([-100, 100])
    ax.legend(loc=1)

    #the statsmodels module does it like this:
    #import statsmodels.api as sm
    #res = sm.tsa.seasonal_decompose(history, freq=7*24)
    #res.plot()
    #plt.show()

    #show plots
    plt.show()

class Predictions:
    """
    A class for predicting gas prices
    """
    
    db = None
    
    def __init__(self):
        """
        Connect to database
        """
        self.db = Database()
        
    def predict_station(self, stid, start=datetime(2016, 5, 3, 0, 0, 0, 0), end=datetime(2017, 3, 19, 0, 0, 0, 0), fuel_type="diesel"):
        
        history = self.db.find_price_hourly_history(stid, start, end, fuel_type)
        plot_split_seasonal(history, predictions=True) 
        
    
if __name__ == "__main__":
    Predictions().predict_station(
        Database().find_stations(place="Strausberg").index[0],
        start=datetime(2017, 1, 19, 0, 0, 0, 0),
        end=datetime(2017, 3, 19, 0, 0, 0, 0)
    )
    