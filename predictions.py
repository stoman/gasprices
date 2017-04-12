from datetime import datetime, timedelta
import multiprocessing

from numpy.random import randint
import pytz
from sklearn import base
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from statsmodels.genmod.cov_struct import Autoregressive
from statsmodels.tsa.stattools import adfuller

from database import Database
from holidays import HolidayTransformer
import numpy as np
import pandas as pd
import seaborn as sbn


sbn.set(font_scale=1.5)

#set default parameters
default_hyperparameters = {
    "p": list(range(1, 5)) + list(range(20, 30)),#number of autoregressive features
    "q": list(range(1, 5)),#number of moving average features
    "r": 4*7*24,#size of history for predicting the residual
    "featureset_trend": ["ar", "ma", "time", "holidays"],
    "featureset_res": ["time", "holidays"],
    "zipcode": 15370#default zipcode is none is found
}

def predict_split(history, prediction_length=7*24, hyperparameters={}):
    """
    This function predicts a time series of gas prices by splitting it into a
    tren and a residual and then applying a feature pipeline and predicting
    each of them individually.

    Keyword arguments:
    history -- the time series to split up
    prediction_length -- the number of time steps to predict (default 7*24)
    hyperparameters -- values used for the prediction model (default {}) 

    Return value:
    2 time series predicted: trend and residual
    """
    #extract parameters
    r = hyperparameters["r"] if "r" in hyperparameters else default_hyperparameters["r"]

    #split data
    trend, res = split_trend(history)
    
    #create index for prediction time series
    index_pred = pd.date_range(
        start=history.index.max() + timedelta(hours=1),
        end=history.index.max() + timedelta(hours=prediction_length),
        freq="1H",
        tz=pytz.utc
    )
    
    #predict the trend    
    trend_pred = predict_ts(
        (trend - trend.shift(1)).fillna(0.),
        get_feature_pipeline("trend", hyperparameters),
        index_pred,
        hyperparameters=hyperparameters
    ).cumsum() + trend.iloc[-1]
    
    #compute residual prediction
    res_pred = predict_ts(
        res.iloc[-r:],
        get_feature_pipeline("res", hyperparameters),
        index_pred,
        hyperparameters=hyperparameters
    )
        
    #alternative: using AR from statsmodels    
    #res_model = AR(res)
    #res_results = res_model.fit(disp=-1, maxlag=p)
    #res_pred = res_results.predict(len(res), len(res) + prediction_length)

    #return result
    return trend_pred, res_pred

def split_trend(history):
    """
    This function splits a time series of gas prices into a trend and a
    residual. The two returned time series sum up to the input time series.

    Keyword arguments:
    history -- the time series to split up

    Return value:
    a tuple of two time series: trend and residual
    """
    #compute trend
    length_week = 7*24
    trend = history.rolling(window=length_week, min_periods=1).mean()
    
    #compute residual
    res = history - trend
    
    return trend, res

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
        r.update({"month%d" % i: t.month == i for i in range(1, 13)})
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
        for i in self.q:
            if i > 1:
                df["ma%05d" % i] = df["ma%05d" % (i - 1)] * (i - 1) / i + X["lag%05d" % i] / i
        return df

class Autoregressive(base.BaseEstimator, base.TransformerMixin):
    """
    An transformer that extracts autoregressive features. It assumes that the
    input is given as a pandas dataframe with columns lag00001, lag00002, ...
    """
    def __init__(self, p):
        self.p = p
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return X[["lag%05d" % i for i in self.p]]

def one_cv_step(i, fold, prediction_length, history, hyperparameters):
    """
    A function for cross-validating the predictions. Can be pickled and used
    within multiprocessing.
    
    Keyword arguments:
    i -- id of the chunk to cross-validate
    fold -- total number of chunks
    prediction_length -- the number of time steps to predict in each prediction
    history -- the history of prices
    
    Return value:
    a tuple of the mean absolute error of the prediction, the mean absolute
    error of a naive prediction, the mean squared error of the prediction, and
    the first time step predicted 
    """
    print("CV fold %d of %d" %(fold - i + 1, fold))
    predictions = predict(
        history.iloc[:-i*prediction_length-1],
        prediction_length=prediction_length,
        hyperparameters=hyperparameters
    )
    
    #compute errors
    ret = (
        (predictions - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1]).abs().mean(),
        (history.iloc[-i*prediction_length-2] - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1]).abs().mean(),
        ((predictions - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1]) ** 2).mean(),
        (history.iloc[-(i+1)*prediction_length-1:-i*prediction_length-1].reset_index(drop=True) - history.iloc[-i*prediction_length-1:-(i-1)*prediction_length-1].reset_index(drop=True)).abs().mean(),
        history.index[-i*prediction_length-1]
    )
    return ret

def predict(history, prediction_length=7*24, hyperparameters={}):
    """
    This function predicts a time series of gas prices.

    Keyword arguments:
    history -- the time series to split up
    prediction_length -- the number of time steps to predict (default 7*24)
    hyperparameters -- values used for the prediction model (default {}) 

    Return value:
    the predicted time series
    """
    #compute predictions of the seasonal split and sum up
    trend_pred, res_pred = predict_split(
        history,
        prediction_length,
        hyperparameters=hyperparameters
    )
    return trend_pred + res_pred     

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
    q = hyperparameters["q"] if "q" in hyperparameters else default_hyperparameters["q"]
    
    #shift and lag ts
    X = pd.DataFrame({"lag%05d" % i: ts.shift(i) for i in p + q})
    y = ts[pd.notnull(X).all(axis=1)]
    X = X[pd.notnull(X).all(axis=1)]
    
    #create pipeline
    pipeline = Pipeline([
        ("features", features),
        #("regressor", LinearRegression(normalize=True))
        ("regressor", RandomForestRegressor())
    ]).fit(X, y)

    #predict data
    ts_all = [y for y in ts]#remove index
    for t in index_pred:
        Xt = pd.DataFrame({"lag%05d" % i: ts_all[-i] for i in p + q}, index=[t])
        ts_all.append(pipeline.predict(Xt)[0])
    ts_pred = pd.Series(ts_all[-len(index_pred):], index=index_pred)
    return ts_pred

def get_feature_pipeline(pipeline_type="trend", hyperparameters={}):
    """
    Create a pipeline assembling all features that we use for predicting
    gas prices from a shifted time series.
    
    Keyword arguments:
    pipeline_type -- the pipeline features to choose, use `featureset_%` from
    hyperparameters (default "trend")
    hyperparameters -- values used for the prediction model (default {})
    
    Return value:
    the sklearn feature pipeline 
    """
    #extract parameters
    featureset = hyperparameters["featureset_%s" % pipeline_type] if "featureset_%s" % pipeline_type in hyperparameters else default_hyperparameters["featureset_%s" % pipeline_type]
    p = hyperparameters["p"] if "p" in hyperparameters else default_hyperparameters["p"]
    q = hyperparameters["q"] if "q" in hyperparameters else default_hyperparameters["q"]
    zipcode = hyperparameters["zipcode"] if "zipcode" in hyperparameters else default_hyperparameters["zipcode"]
    
    #assemble pipeline
    pipeline = []
    if "ar" in featureset:
        pipeline.append(("ar", Autoregressive(p)))
    if "ma" in featureset:
        pipeline.append(("ma", MovingAverage(q)))
    if "time" in featureset:
        pipeline.append(("time", DateFeatures()))
    if "holidays" in featureset:
        pipeline.append(("holidays", HolidayTransformer(zipcode=zipcode)))
    return FeatureUnion(pipeline)


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
            predictions=True,
            cv=True,
            prediction_length=prediction_length,
            hyperparameters={"zipcode": station.iloc[0]["post_code"]}
        ) 
        
    def cross_validation(self, stid, start=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc), end=datetime(2017, 3, 19, 0, 0, 0, 0, pytz.utc), fuel_type="diesel", prediction_length=7*24, fold=52):
        #compute price history
        history = self.db.find_price_hourly_history(stid, start=start, end=end, fuel_type=fuel_type)
        fold = min(fold, len(history) // prediction_length - 1)
        station = self.db.find_stations(stids=[stid])
        
        #compute errors for some past time frames
        errors = multiprocessing.Pool().starmap(one_cv_step, [(i, fold, prediction_length, history, {"zipcode": station.iloc[0]["post_code"]}) for i in range(fold, 0, -1)])
        abs_errors, naive_error, mse, shift24, index = zip(*errors)    

        #create dataframe to return
        return pd.DataFrame({
            "date": index,
            "absolute": abs_errors,
            "mse": mse,
            "last": naive_error,
            "shift 24": shift24,
            "r2 (last)": [1. - a / n if not n == 0 else -1 for a, n in zip(abs_errors, naive_error)],
            "r2 (shift 24)": [1. - a / n if not n == 0 else -1 for a, n in zip(abs_errors, shift24)],
        })
                
if __name__ == "__main__":
    #usage samples
    n = 10
    stations = Database().find_stations(active_after=datetime(2017, 3, 1, 0, 0, 0, 0, pytz.utc), active_before=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc))
    print("selecting %d out of %d valid gas stations" % (n, len(stations)))
    np.random.seed(42)
    stids = stations.index[randint(0, len(stations), n)]
    errors = pd.DataFrame()
    for i, stid in enumerate(stids):
        print("station %d of %d" % (i + 1, len(stids)))
        new_errors = Predictions().cross_validation(stid, fold=8, prediction_length=7*24)
        new_errors["stid"] = stid
        errors = errors.append(new_errors)
        print(new_errors.describe())
    print("All errors")
    print(errors.describe())
    print("Errors by date")
    print(errors.groupby("date").mean())
    print("Errors by station")
    print(errors.groupby("stid").mean())
