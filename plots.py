from datetime import datetime, timedelta

import pytz

from database import Database
import matplotlib.pyplot as plt
import pandas as pd
from predictions import split_seasonal, predict_split, Predictions
import seaborn as sbn


sbn.set(font_scale=1.5)


class Plots:
    """
    A class for creating all kinds of plots
    """
    db = None
    
    def __init__(self):
        """
        Connect to database
        """
        self.db = Database()
        
    def brands(self):
        """
        Plot a pie chart of gas station brands
        """
        #create pandas dataframe
        brands = self.db.find_stations().groupby("brand")["brand"].count()
        brands = brands.rename("Count of Gas Stations", inPlace=True)
        commonbrands = brands[brands >= brands.sum()/40]
        commonbrands.set_value("Others", brands.sum() - commonbrands.sum())
        
        #create plot
        commonbrands.sort_values().plot(
            kind="pie",
            title="Number of Gas Stations by Brand",
            autopct="%1.0f%%"
        )
        fig = plt.gcf()
        fig.gca().add_artist(plt.Circle((0, 0), 0.7, fc="white"))
    
    def prices(self, stids=[], start=pytz.utc.localize(datetime.utcnow()) - timedelta(days=14), end=pytz.utc.localize(datetime.utcnow()), title="", nightstart=22, nightend=6, fuel_types=["diesel", "e5", "e10"]):
        """
        Plot a line chart containing the average price history of some gas
        stations.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations (default
        [])
        start -- the first update time to include in the price history (default
        datetime.now() - timedelta(days=14)
        end -- the last update time to include in the price history (default
        datetime.now())
        title -- title of the diagram (default "")
        nightstart -- first hour of the day to highlight as night time (default
        22)
        nightend -- last hour of the day to highlight as night time (default 6)
        fuel_types -- a list of fuel types to plot (default ["diesel", "e5",
        "e10"])
        """
                                 
        #query for initial values
        current_prices = self.db.find_prices(stids, time=start, fuel_types=fuel_types)

        #query database for price history
        history = self.db.find_price_history(stids=stids, start=start, end=end)

        #save mean prices
        mean_prices = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}).set_index("date")
        for fuel_type in fuel_types:
            mean_prices[fuel_type] = .0
    
        #read price changes
        for date, change in history.iterrows():
            for fuel_type in fuel_types:
                #update price only if there is no obvious mistake
                if change[fuel_type]:
                    current_prices.loc[change["stid"]][fuel_type] = change[fuel_type]
                #insert or update row in mean prices
                if date in mean_prices.index:
                    mean_prices.update(pd.DataFrame(current_prices.mean().set_value("date", date)))
                else:
                    mean_prices.loc[date] = current_prices.mean()


        if end in mean_prices.index:
            mean_prices.update(pd.DataFrame(current_prices.mean().set_value("date", end)))
        else:
            mean_prices.loc[end] = current_prices.mean().set_value("date", end)

        #create the plot
        ax = mean_prices.plot(drawstyle="steps", title=title)
        ax.set_ylabel("Price in 1/1000 Euros")
        ax.set_xlabel("Date")
    
        #highlight night times
        mindate = history.index.min()
        startdate = datetime(mindate.year, mindate.month, mindate.day) - timedelta(days=1)
        maxdate = history.index.max()
        enddate = datetime(maxdate.year, maxdate.month, maxdate.day) + timedelta(days=1)
        while startdate < enddate:
            ax.axvspan(
                startdate + timedelta(hours=nightstart),
                startdate + timedelta(hours=24 + nightend),
                facecolor="g" if startdate.weekday() < 5 else "b",
                alpha=0.2
            )
            startdate += timedelta(days=1)
            
        #show plot
        ax.legend(bbox_to_anchor=(1., .95))

    def split(self, history, features, predictions=False, prediction_length=7*24, cv=True):
        """
        This function plots an overview over the seasonal split of a history of gas
        prices as computed by `predictions.split_seasonal`.
        
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
            trend_pred, weekly_pred, res_pred = predict_split(
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

#sample usage
if __name__ == "__main__":
    plots = Plots()
    database = Database()
    
    #plot pie chart of brands
    plots.brands()
    
    #plot prices in a city
    plots.prices(
        stids=database.find_stations(place="Strausberg").index.tolist(),
        title="Fuel Prices in Strausberg",
        start=pytz.utc.localize(datetime.utcnow()) - timedelta(weeks=5),
        end=pytz.utc.localize(datetime.utcnow()) - timedelta(weeks=2)
    )
    
    #plot seasonal split
    db = Database()
    station = db.find_stations(place="Strausberg")
    history = db.find_price_hourly_history(station.index[0])
    plots.split(
        history,
        Predictions().get_feature_pipeline(zipcode=station.iloc[0]["post_code"]),
        predictions=True,
        cv=True
    )
    
    #show plots
    plt.show()
