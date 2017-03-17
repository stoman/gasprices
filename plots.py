from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

from database import Database

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
        ax = commonbrands.sort_values().plot(
            kind="pie",
            title="Number of Gas Stations by Brand",
            autopct="%1.0f%%"
        )
        fig = plt.gcf()
        fig.gca().add_artist(plt.Circle((0, 0), 0.7, fc="white"))
        plt.show()
    
    def prices(self, stids=[], start=datetime.now() - timedelta(days=14), end=datetime.now(), title="", nightstart=22, nightend=6, fuel_types=["diesel"]):
        """
        Plot a line chart containing the average price history of some gas stations.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations (default [])
        start -- the first update time to include in the price history (default datetime.now() - timedel
        end -- the last update time to include in the price history (default datetime.now())
        title -- title of the diagram (default "")
        nightstart -- first hour of the day to highlight as night time (default 22)
        nightend -- last hour of the day to highlight as night time (default 6)
        fuel_types -- a list of fuel types to plot (default ["diesel"])
        """
                                 
        #query database for price history
        history = self.db.find_prices(stids=stids, start=start, end=end)

        #fix me: initial values
        current_prices = pd.DataFrame(
            [{"stid": stid}.update({fuel_type: .0 for fuel_type in fuel_types}) for stid in stids]
        ).set_index("stid")

        #save mean prices
        mean_prices = pd.DataFrame({
                "date": pd.Series(dtype="datetime64[ns]")
            }.update({fuel_type: pd.Series(dtype="float64") for fuel_type in fuel_types})
        ).set_index("date")
    
        #read price changes
        for _, change in history.iterrows():
            for fuel_type in fuel_types:
                current_prices.loc[change["stid"]][fuel_type] = change[fuel_type]
                mean_prices.loc[change["date"]] = current_prices.mean().set_value("date", change["date"])
    
        #create the plot
        ax = mean_prices.plot(drawstyle="steps", title=title)
        ax.set_ylabel("Price in 1/1000 Euros")
        ax.set_xlabel("Date")
    
        #highlight night times
        mindate = history["date"].min()
        startdate = datetime(mindate.year, mindate.month, mindate.day) - timedelta(days=1)
        maxdate = history["date"].max()
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
        plt.show()

#sample usage
if __name__ == "__main__":
    plots = Plots()
    database = Database()
    
    #plot pie chart of brands
    #plots.brands()
    
    #plot prices in a city
    plots.prices(
        stids=database.find_stations(place="Kassel")["id"].tolist(),
        title="Fuel Prices in Kassel"
    )
