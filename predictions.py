from datetime import datetime, timedelta
import multiprocessing
from pickletools import optimize

from numpy.random import randint
import pytz
from sklearn import base
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics.regression import r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from statsmodels.tsa.arima_model import ARMA, AR
from statsmodels.tsa.stattools import adfuller, acf, pacf

from database import Database
from holidays import HolidayTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn


sbn.set(font_scale=1.5)

#set default parameters
default_hyperparameters = {
    "p": 2*24,
    "q": 2*24,
    "featureset": ["ar", "ma", "time", "holidays"]
}

def predict_split(history, features, prediction_length=7*24, hyperparameters={}):
    """
    This function predicts a time series of gas prices by splitting it into a
    trend, a weekly pattern, and a residual and then applying a feature
    pipeline and predicting each of them individually.

    Keyword arguments:
    history -- the time series to split up
    features -- a pipeline to transform the features before predicting them
    prediction_length -- the number of time steps to predict (default 7*24)
    hyperparameters -- values used for the prediction model (default {}) 

    Return value:
    3 time series predicted: trend, weekly pattern, and residual
    """
    #split data
    trend, weekly, res = split_seasonal(history)
    
    #create index for prediction time series
    index_pred = pd.date_range(
        start=history.index.max() + timedelta(hours=1),
        end=history.index.max() + timedelta(hours=prediction_length),
        freq="1H",
        tz=pytz.utc
    )
    
    #compute weekly prediction
    weekly_pred = pd.Series(index=index_pred)
    length_week = 7*24
    for i in range(length_week):
        weekly_pred[i::length_week] = weekly[-length_week + i]

    #Analyze trend
    #trend_shift = (trend - trend.shift(1)).fillna(0.)
    #print("Shifted Trend")
    #test_stationary(trend_shift)
    #plot acf to debug
    #plt.plot(acf(trend_shift))
    #plt.show()

    #predict the trend    
    trend_pred = predict_ts(trend, features, index_pred, hyperparameters=hyperparameters)

    #alternative: using AR from statsmodels    
    #trend_model = AR(trend_shift)
    #trend_results = trend_model.fit(disp=-1, maxlag=p)
    #trend_res = trend_results.predict(len(trend), len(trend) + prediction_length)
    #trend_pred = trend_res.cumsum() + trend[-1]
    
    #compute residual prediction
    #res_pred = self.predict_ts(res, features, index_pred, hyperparameters=hyperparameters)
    
    #alternative: set zero
    res_pred = pd.Series(index=trend_pred.index).fillna(0.)
    
    #alternative: using AR from statsmodels    
    #res_model = AR(res)
    #res_results = res_model.fit(disp=-1, maxlag=p)
    #res_pred = res_results.predict(len(res), len(res) + prediction_length)

    #return result
    return trend_pred, weekly_pred, res_pred

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
    """
    A transformer that creates features for the hour, day of week, and more
    of the index of a time series using a `DictVectorizer`.
    """
    def fit(self, X, y=None):
        return self
    
    def compute_row(self, t):
        r = {}
        r.update({"hour%d" % i: t.hour == i for i in range(24)})
        r.update({"dow%d" % i: t.dayofweek == i for i in range(7)})
        #r.update({"week%d" % i: t.weekofyear == i for i in range(53)})
        return r

    def transform(self, X):
        features = [self.compute_row(t) for t in X.index]
        matrix = DictVectorizer().fit_transform(features)
        return matrix.todense()

class MovingAverage(base.BaseEstimator, base.TransformerMixin):
    """
    An transformer that computes moving averages. It assumes that the
    input is given as a pandas dataframe with columns lag00001, lag00002, ...
    only and returns a pandas dataframe with columns ma00001, ma00002, ...
    """
    def __init__(self, q):
        self.q = q
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        df = pd.DataFrame({"ma%05d" % 1: X["lag%05d" % 1]}, index=X.index)
        for i in range(2, len(X.columns) + 1):
            df["ma%05d" % i] = df["ma%05d" % (i - 1)] * (i - 1) / i + X["lag%05d" % i] / i
        return df

def one_cv_step(i, fold, prediction_length, history, pipeline):
    """
    A function for cross-validating the predictions. Can be pickled and used
    within multiprocessing.
    
    Keyword arguments:
    i -- id of the chunk to cross-validate
    fold -- total number of chunks
    prediction_length -- the number of time steps to predict in each prediction
    history -- the history of prices
    features -- a pipeline creating additional features to use while predicting
    
    Return value:
    a tuple of the mean absolute error of the prediction, the mean absolute
    error of a naive prediction, the mean squared error of the prediction, and
    the first time step predicted 
    """
    print("CV fold %d of %d" %(fold - i + 1, fold))
    predictions = predict(
        history.iloc[:-i*prediction_length-1],
        pipeline,
        prediction_length=prediction_length
    )
    
    #compute errors
    ret = (
        (predictions - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1]).abs().mean(),
        (history.iloc[-i*prediction_length-2] - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1]).abs().mean(),
        ((predictions - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1]) ** 2).mean(),
        history.index[-i*prediction_length-1]
    )
    return ret

def predict(history, features, prediction_length=7*24, hyperparameters={}):
    """
    This function predicts a time series of gas prices.

    Keyword arguments:
    history -- the time series to split up
    features -- a pipeline to transform the features before predicting them
    prediction_length -- the number of time steps to predict (default 7*24)
    hyperparameters -- values used for the prediction model (default {}) 

    Return value:
    the predicted time series
    """
    #compute predictions of the seasonal split and sum up
    trend_pred, weekly_pred, res_pred = predict_split(history, features, prediction_length, hyperparameters=hyperparameters)
    return trend_pred + weekly_pred + res_pred     

def predict_ts(ts, features, index_pred, hyperparameters={}):
    """
    Predict a time series related to fuel prices. This could be used for the
    trend or the residual.
    
    Keyword arguments:
    ts -- a time series to predict
    features -- a pipeline to transform the features before predicting them
    index_pred -- the index of the values to predict. Indices are used for
    additional features, but assigned one after another independent of the
    time span between them.
    hyperparameters -- values used for the prediction model (default {}) 
    """
    #extract parameters
    p = hyperparameters["p"] if "p" in hyperparameters else default_hyperparameters["p"]
    
    #shift and lag ts
    ts_shift = (ts - ts.shift(1)).fillna(0.)
    X = pd.DataFrame({"lag%05d" % i: ts_shift.shift(i) for i in range(1, p + 1)[::-1]}).iloc[p:]
    y = ts_shift.iloc[p:]
    
    #create pipeline
    pipeline = Pipeline([
        ("features", features),
        ("regressor", LinearRegression(normalize=True))
    ]).fit(X, y)

    #predict data
    ts_all = [y for y in ts_shift]#remove index
    for t in index_pred:
        Xt = pd.DataFrame({"lag%05d" % i: ts_all[-i] for i in range(1, p + 1)[::-1]}, index=[t])
        ts_all.append(pipeline.predict(Xt)[0])
    ts_pred = pd.Series(ts_all[-len(index_pred):], index=index_pred)
    return ts[-1] + ts_pred.cumsum()

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
        
    def predict_station(self, stid, start=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc), end=datetime(2017, 3, 19, 0, 0, 0, 0, pytz.utc), fuel_type="diesel", prediction_length=7*24):
        history = self.db.find_price_hourly_history(stid, start=start, end=end, fuel_type=fuel_type)
        station = self.db.find_stations(stids=[stid])
        self.plot_split_seasonal(
            history,
            self.get_feature_pipeline(zipcode=station.iloc[0]["post_code"]),
            predictions=True,
            cv=True,
            prediction_length=prediction_length
        ) 
        
    def cross_validation(self, stid, start=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc), end=datetime(2017, 3, 19, 0, 0, 0, 0, pytz.utc), fuel_type="diesel", prediction_length=7*24, fold=52):
        #compute price history
        history = self.db.find_price_hourly_history(stid, start=start, end=end, fuel_type=fuel_type)
        fold = min(fold, len(history) // prediction_length - 1)
        station = self.db.find_stations(stids=[stid])
        pipeline = self.get_feature_pipeline(zipcode=station.iloc[0]["post_code"])
        
        #compute errors for some past time frames
        errors = multiprocessing.Pool().starmap(one_cv_step, [(i, fold, prediction_length, history, pipeline) for i in range(fold, 0, -1)])
        abs_errors, naive_error, mse, index = zip(*errors)    

        #create dataframe to return
        return pd.DataFrame({
                "absolute": abs_errors,
                "mse": mse,
                "naive": naive_error,
                "r2": [1. - a / n if not n == 0 else -999 for a, n in zip(abs_errors, naive_error)]
            },
            index=index
        )
    
    def get_feature_pipeline(self, zipcode, hyperparameters={}):
        """
        Create a pipeline assembling all features that we use for predicting
        gas prices from a shifted time series.
        
        Keyword arguments:
        zipcode -- the zipcode of the gas station (needed for finding public
        holidays)
        hyperparameters -- values used for the prediction model (default {})
        
        Return value:
        the sklearn feature pipeline 
        """
        #extract parameters
        featureset = hyperparameters["featureset"] if "featureset" in hyperparameters else default_hyperparameters["featureset"]
        q = hyperparameters["q"] if "q" in hyperparameters else default_hyperparameters["q"]
        
        #assemble pipeline
        pipeline = []
        if "ar" in featureset:
            pipeline.append(("ar", FunctionTransformer()))#identity function for first p features
        if "ma" in featureset:
            pipeline.append(("ma", MovingAverage(q)))
        if "time" in featureset:
            pipeline.append(("time", DateFeatures()))
        if "holidays" in featureset:
            pipeline.append(("holidays", HolidayTransformer(zipcode=zipcode)))
        return FeatureUnion(pipeline)
        
    def plot_split_seasonal(self, history, features, predictions=False, prediction_length=7*24, cv=True):
        """
        This function plots an overview over the seasonal split of a history of gas
        prices as computed by `split_seasonal`.
        
        Keyword arguments:
        history -- the time series to split up and visualize
        features -- a pipeline to transform the features before predicting them
        predictions -- whether to plot predictions too (default False)
        prediction_length -- number of time steps to predict. This is only
        effective if `predictions` is set to `True` (default 4*7*24)
        cv -- whether to predict data that is known and can be cross-validated
        (default True)   
        """
        #split price history
        trend, weekly, res = split_seasonal(history)
    
        #predict price history
        if predictions:
            trend_pred, weekly_pred, res_pred = self.predict_split(
                history[:-prediction_length] if cv else history,
                features,
                prediction_length=prediction_length
            )
            history_pred = trend_pred + weekly_pred + res_pred
            
            #print quality of predictions
            #prediction_error = history[-prediction_length:] - history_pred
            #print("Mean absolute prediction error %f" % prediction_error.abs().mean())
    
        #run Dickey-Fuller test for debugging
        #print("History Without Trend")
        #test_stationary(history - trend)
        #print("Residual")
        #test_stationary(res)
    
        #plot given time series
        palette = sbn.color_palette(n_colors=6)
        ax = plt.subplot(3, 1, 1) 
        plt.plot(history, label="Price History", color=palette[1])
        plt.plot(trend, label="Trend", color=palette[2])
        if predictions:
            ax.axvline(history_pred.index[0])
            plt.plot(history_pred, linestyle="dashed", color=palette[1])
            plt.plot(trend_pred, linestyle="dashed", color=palette[2])
        ax.legend(loc=1)
            
        ax = plt.subplot(3, 1, 2)
        plt.plot(history - trend, label="Price History without Trend", color=palette[3])
        plt.plot(weekly, label="Weekly Pattern", color=palette[4])
        if predictions:
            ax.axvline(history_pred.index[0])
            plt.plot(history_pred - trend_pred, linestyle="dashed", color=palette[3])
            plt.plot(weekly_pred, linestyle="dashed", color=palette[4])
        ax.set_ylim([-100, 100])
        ax.legend(loc=1)
    
        ax = plt.subplot(3, 1, 3)
        plt.plot(res, label="Residual", color=palette[5])
        if predictions:
            ax.axvline(history_pred.index[0])
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
    
if __name__ == "__main__":
    #usage samples
#     Predictions().predict_station(
#         Database().find_stations(place="Strausberg").index[0],
#         #start=datetime(2017, 2, 1, 0, 0, 0, 0, pytz.utc),
#         end=datetime(2017, 1, 29, 0, 0, 0, 0, pytz.utc)
#     )
    n = 10
    stations = Database().find_stations(active_after=datetime(2017, 3, 1, 0, 0, 0, 0, pytz.utc), active_before=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc))
    print("selecting %d out of %d valid gas stations" % (n, len(stations)))
    np.random.seed(42)
    stids = stations.index[randint(0, len(stations), n)]
    errors = pd.DataFrame()
    for i, stid in enumerate(stids):
        print("station %d of %d" % (i + 1, len(stids)))
        new_errors = Predictions().cross_validation(stid)
        new_errors["stid"] = stid
        errors = errors.append(new_errors)
    print(errors.describe())
    print(errors.groupby("date").mean().describe())
    print(errors.groupby("stid").mean().describe())
