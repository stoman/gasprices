from database import Database
import matplotlib.pyplot as plt
import seaborn as sbn

class Plots:
    """A class for creating all kinds of plots"""
    
    db = None
    
    def __init__(self):
        """Connect to database"""
        self.db = Database()
        
    def brands(self):
        """Plot a pie chart of gas station brands"""
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

if __name__ == "__main__":
    plots = Plots()
    plots.brands()