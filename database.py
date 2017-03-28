from datetime import datetime, timedelta
import pandas as pd
import pytz
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import func
import urllib
import io
import gzip

class Database:
    """
    A class representing the connection to the gas prices database
    """
    
    connection = None
    meta = None
    session = None
    
    def __init__(self, dbconfig=None):
        """
        Connect to the database using a config object containing the database
        credentials.
        
        If no `configdb` argument is given we use the credentials from the file
        `db.ini`

        Keyword arguments:
        configdb -- a `SafeConfigParser` containing database credentials
        (default None)
        """
        #find database credentials
        if not dbconfig:
            dbconfig = SafeConfigParser()
            dbconfig.read("db.ini")
        
        #connect to database
        self.connection = sqlalchemy.create_engine(
            "{type}://{user}:{passwd}@{host}:{port}/{dbname}".format(
                type=dbconfig.get("database", "type"),
                user=dbconfig.get("database", "user"),
                passwd=dbconfig.get("database", "passwd"),
                host=dbconfig.get("database", "host"),
                port=dbconfig.get("database", "port"),
                dbname=dbconfig.get("database", "dbname")
            ), client_encoding="utf8"
        )
        self.meta = sqlalchemy.MetaData(bind=self.connection, reflect=True)
        Session = sessionmaker(bind=self.connection)
        self.session = Session()
    
    def find_stations(self, place=None):
        """
        Create a pandas dataframe containing all gas stations with the given
        properties.
        
        Keyword arguments:
        place -- the city of the gas stations (default None)
        
        Return value:
        a pandas dataframe containing all information about the gas stations
        from the database
        """
        #construct query
        table = self.meta.tables["gas_station"]
        query = self.session.query(table)
        if place:
            query = query.filter(table.c.place == place)
        
        #create pandas dataframe
        return pd.read_sql(query.statement, self.connection).set_index("id")
        
    def find_price_history(self, stids, start=pytz.utc.localize(datetime.utcnow()) - timedelta(days=14), end=pytz.utc.localize(datetime.utcnow())):
        """
        Create a pandas dataframe containing all price changes with the given
        properties.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations
        start -- the first update time to include in the price history (default
        datetime.now() - timedelta(days=14))
        end -- the last update time to include in the price history (default
        datetime.now())

        Return value:
        a pandas dataframe containing all information about the price changes
        from the database
        """
        #construct query
        table = self.meta.tables["gas_station_information_history"]
        query = self.session.query(table).filter(sqlalchemy.and_(
            table.c.date >= start, table.c.date <= end, table.c.stid.in_(stids)
        )).order_by(table.c.date)
        
        #create pandas dataframe
        df = pd.read_sql(query.statement, self.connection)
        df["date"] = [pytz.utc.localize(t) for t in pd.to_datetime(df["date"], utc=True)]
        return df.set_index("date")
        
    def find_price_hourly_history(self, stid, start=pytz.utc.localize(datetime.utcnow()) - timedelta(days=14), end=pytz.utc.localize(datetime.utcnow()), fuel_type="diesel"):
        """
        Create a pandas series containing the hourly price at a given gas
        station.
        
        Keyword arguments:
        stid -- the id of a gas stations
        start -- the first update time to include in the price history (default
        datetime.now() - timedelta(days=14))
        end -- the last update time to include in the price history (default
        datetime.now())
        fuel_type -- the type of fuel to find prices for (default "diesel")
        
        Return value:
        a pandas dataframe containing the hourly prices at the gas station
        """
        #get historic price data from database
        history_compressed = self.find_price_history([stid], start=start, end=end)[fuel_type]
        history_compressed = history_compressed[history_compressed > 0]
        
        #get opening price from database
        current_price = self.find_prices([stid], time=start, fuel_types=[fuel_type])[fuel_type][0]

        #uncompress price data
        history = pd.Series()
        index_compressed = 0
        for date in pd.date_range(start=start, end=end, freq="1H", tz=pytz.utc):
            if index_compressed < len(history_compressed) and history_compressed.index[index_compressed] <= date:
                current_price = history_compressed.iloc[index_compressed]
                index_compressed += 1
            history[date] = current_price

        return history

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close connection to the database
        """
        self.connection.close()
        
    def import_database(self, url="https://creativecommons.tankerkoenig.de/history/history.dump.gz"):
        """
        Imports a gzipped database file from an external server. Make sure you
        trust this server!
        
        Keyword arguments:
        url -- the url of the file to load (default
        "https://creativecommons.tankerkoenig.de/history/history.dump.gz")
        """
        response = urllib.request.urlopen(url)
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        self.connection.execute(decompressed_file.read())
        
    def create_indices(self):
        """
        Create some indices on the database for faster queries
        """
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_date_stid ON gas_station_information_history USING btree(date, stid);"
        )
        
    def find_prices(self, stids, time=pytz.utc.localize(datetime.utcnow()), fuel_types=["diesel", "e5", "e10"]):
        """
        Create a pandas dataframe containing the current prices at the given
        gas stations and the given time.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations
        time -- the time to find the gas prices (default datetime.now())
        fuel_types -- a list of fuel types to return (default ["diesel", "e5",
        "e10"])

        Return value:
        a pandas dataframe containing the current prices of the given gas
        stations
        """
        #construct sub query with time of last change
        table = self.meta.tables["gas_station_information_history"]
        date_query = self.session.query(
            func.max(table.c.date), table.c.stid
        ).filter(
            sqlalchemy.and_(
                table.c.date <= time, table.c.stid.in_(stids)
            )
        ).group_by(table.c.stid).subquery()
        
        #construct actual query
        query = self.session.query(table).join(
            date_query,
            sqlalchemy.and_(
                table.c.stid == date_query.c.stid,
                table.c.date == list(date_query.c)[0]
            )
        )
        
        #create pandas dataframe
        return pd.read_sql(query.statement, self.connection)[["stid"] + fuel_types].set_index("stid")
    
#sample usage    
from configparser import SafeConfigParser
if __name__ == "__main__":
    db = Database()
    print(db.find_stations(place="Kassel"))
