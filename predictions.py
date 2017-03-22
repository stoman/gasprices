from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from database import Database

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
        
    def predict_station(self, stid, fuel_type="diesel", start=datetime(2016, 5, 3, 0, 0, 0, 0), end=datetime(2017, 3, 19, 0, 0, 0, 0)):
        
        #get historic price data from database
        history_compressed = self.db.find_price_history([stid], start=start, end=end)[fuel_type]
        history_compressed = history_compressed[history_compressed > 0]
        
        #uncompress price data
        history = pd.Series()
        index_compressed = 0
        current_price = self.db.find_prices([stid], time=start, fuel_types=[fuel_type])[fuel_type][0]
        for date in pd.date_range(start=start, end=end, freq="1H"):
            if index_compressed < len(history_compressed) and history_compressed.index[index_compressed] <= date:
                current_price = history_compressed.iloc[index_compressed]
                index_compressed += 1
            history[date] = current_price
        
        #compute rolling average
        length_week = 7*24
        trend = history.rolling(window=length_week, center=True, min_periods=1).mean()
        res = (history - trend).dropna()
        weekly = [res[-length_week+i::-length_week].mean() for i in range(length_week)]
        seasonal = np.tile(weekly, 1 + len(history) // length_week)
        seasonal = seasonal[len(seasonal) - len(res):]
        seasonal = pd.Series(seasonal, index=res.index)
        res -= seasonal
        flatt_res = res.rolling(window=3, center=True, min_periods=1).mean()
        
        #plot time series
        ax = plt.subplot(4, 1, 1)
        plt.plot(history, label="Price History")
        plt.plot(trend, label="Trend")
        ax.legend(loc=1)
        ax = plt.subplot(4, 1, 2)
        plt.plot((history - trend), label="Price History without Trend")
        plt.plot(seasonal, label="Weekly Schema")
        ax.set_ylim([-100, 100])
        ax.legend(loc=1)
        ax = plt.subplot(4, 1, 3)
        plt.plot(res, label="Residual")
        plt.plot(flatt_res, label="Flattened Residual")
        ax.set_ylim([-100, 100])
        ax.legend(loc=1)
        ax = plt.subplot(4, 1, 4)
        plt.plot((res - flatt_res), label="Remaining Noise")
        ax.set_ylim([-100, 100])
        ax.legend(loc=1)
        plt.show()
        
        res = sm.tsa.seasonal_decompose(history, freq=length_week)
        res.plot()
        plt.show()
    
if __name__ == "__main__":
    Predictions().predict_station(
        Database().find_stations(place="Strausberg").index[0],
        #start=datetime.now() - timedelta(weeks=7),
        #end=datetime.now() - timedelta(weeks=1)
    )
    