from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from sklearn import base
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from statsmodels.tsa.arima_model import ARMA, AR
from statsmodels.tsa.stattools import adfuller, acf, pacf
import seaborn as sbn
sbn.set(font_scale=1.5)

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
        
class DateFeatures(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def compute_row(self, t):
        r = {}
        #r.update({"hour_%d" % i: t.hour == i for i in range(24)})
        r.update({"dow_%d" % i: t.dayofweek == i for i in range(7)})
        return r

    def transform(self, X):
        features = [self.compute_row(t) for t in X.index]
        matrix = DictVectorizer().fit_transform(features)
        return matrix.todense()

def predict_trend(trend, index_pred, p=2*7*24):
    #shift and lag trend
    trend_shift = (trend - trend.shift(1)).fillna(0.)
    X = pd.DataFrame({"lag%05d" % i: trend_shift.shift(i) for i in range(1, p + 1)}).iloc[p:]
    y = trend_shift.iloc[p:]
    
    #create pipeline
    features = FeatureUnion([
        ("lagged", FunctionTransformer()),
        ("time", DateFeatures())
    ])
    pipeline = Pipeline([
        ("features", features),
        #("squares", PolynomialFeatures(degree=2)),
        ("regressor", LinearRegression(normalize=True))
    ]).fit(X, y)
    
    #predict data
    trend_all = [y for y in trend_shift]#remove index, this is NOT in place
    for t in index_pred:
        Xt = pd.DataFrame({"lag%05d" % i: trend_all[-i] for i in range(1, p + 1)}, index=[t])
        trend_all.append(pipeline.predict(Xt)[0])
    trend_pred = pd.Series(trend_all[-len(index_pred):], index=index_pred)
    return trend[-1] + trend_pred.cumsum()

def predict(history, hours=2*4*7*24, trend=None, weekly=None, res=None):
    """
    This function splits a time series of gas prices into a trend, a weekly
    pattern, and a residual. The three returned time series sum up to the
    input time series.

    Keyword arguments:
    history -- the time series to split up

    Return value:
    a tuple of three time series: trend, weekly, and residual
    """
    #split data if not already given
    if trend is None or weekly is None or res is None:
        trend, weekly, res = split_seasonal(history)
    
    #create index for prediction time series
    index_pred = pd.date_range(
        start=history.index.max() + timedelta(hours=1),
        end=history.index.max() + timedelta(hours=hours),
        freq="1H",
        tz=pytz.utc
    )
    
    #compute weekly prediction
    weekly_pred = pd.Series(index=index_pred)
    length_week = 7*24
    for i in range(length_week):
        weekly_pred[i::length_week] = weekly[-length_week + i]

    #compute trend prediction
    trend_shift = (trend - trend.shift(1)).fillna(0.)
    
    print("Shifted Trend")
    test_stationary(trend_shift)
    #plot acf to debug
    #plt.plot(acf(trend_shift))
    #plt.show()
    
    trend_pred = predict_trend(trend, index_pred)

    #alternative: using AR from statsmodels    
    #trend_model = AR(trend_shift)
    #trend_results = trend_model.fit(disp=-1, maxlag=7*24)
    #trend_res = trend_results.predict(len(trend), len(trend) + hours)
    #trend_pred = trend_res.cumsum() + trend[-1]
    
    #compute residual prediction
    res_pred = predict_trend(res, index_pred)
    
    #alternative: set zero
    #res_pred = pd.Series(index=trend_pred.index).fillna(0.)
    
    #alternative: using AR from statsmodels    
    #res_model = AR(res)
    #res_results = res_model.fit(disp=-1, maxlag=7*24)
    #res_pred = res_results.predict(len(res), len(res) + hours)

    #return result
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

    print("History Without Trend")
    test_stationary(history - trend)
    print("Residual")
    test_stationary(res)


    #plot given time series
    palette = sbn.color_palette(n_colors=6)
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
        
    def predict_station(self, stid, start=datetime(2016, 5, 3, 0, 0, 0, 0, pytz.utc), end=datetime(2017, 3, 19, 0, 0, 0, 0, pytz.utc), fuel_type="diesel"):
        history = self.db.find_price_hourly_history(stid, start=start, end=end, fuel_type=fuel_type)
        plot_split_seasonal(history, predictions=True) 
        
    
if __name__ == "__main__":
    Predictions().predict_station(
        Database().find_stations(place="Strausberg").index[0],
        start=datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc),
        end=datetime(2017, 3, 19, 0, 0, 0, 0, pytz.utc)
    )
    