from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pytz


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

def plot_split_seasonal(history):
    """
    This function plots an overview over the seasonal split of a history of gas
    prices as computed by `split_seasonal`.
    
    Keyword arguments:
    history -- the time series to split up and visualize
    """
    #split price history
    trend, weekly, res = split_seasonal(history)
    print((history - trend).abs().mean())        
    print(res.abs().mean())        
    
    #plot time series
    ax = plt.subplot(3, 1, 1)
    plt.plot(history, label="Price History")
    plt.plot(trend, label="Trend")
    ax.legend(loc=1)
    ax = plt.subplot(3, 1, 2)
    plt.plot(history - trend, label="Price History without Trend")
    plt.plot(weekly, label="Weekly Scheme")
    ax.set_ylim([-100, 100])
    ax.legend(loc=1)
    ax = plt.subplot(3, 1, 3)
    plt.plot(res, label="Residual")
    ax.set_ylim([-100, 100])
    ax.legend(loc=1)
    plt.show()

    #the statsmodels module does it like this:
    #import statsmodels.api as sm
    #res = sm.tsa.seasonal_decompose(history, freq=7*24)
    #res.plot()
    #plt.show()


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
        plot_split_seasonal(history) 
        
    
if __name__ == "__main__":
    Predictions().predict_station(
        Database().find_stations(place="Strausberg").index[0],
        start=datetime(2017, 1, 19, 0, 0, 0, 0, pytz.utc),
        end=datetime(2017, 3, 19, 0, 0, 0, 0, pytz.utc)
    )
    