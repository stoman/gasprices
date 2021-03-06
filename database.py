from configparser import SafeConfigParser
from datetime import datetime, timedelta
import gzip
import io
import urllib

import pytz
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import func

import pandas as pd


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
            dbconfig.read("config.ini")
        
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
    
    def find_stations(self, place=None, stids=None, active_after=None, active_before=None):
        """
        Create a pandas dataframe containing all gas stations with the given
        properties.
        
        Keyword arguments:
        place -- the city of the gas stations (default None)
        stids -- a list of station ids to filter for
        
        Return value:
        a pandas dataframe containing all information about the gas stations
        from the database
        """
        #construct query
        table = self.meta.tables["gas_station"]
        query = self.session.query(table)
        if place:
            query = query.filter(table.c.place == place)
        if stids:
            query = query.filter(table.c.id.in_(stids))
        
        if active_after or active_before:
            #construct subquery for first and last price change
            price_table = self.meta.tables["gas_station_information_history"]
            sq = self.session.query(
                price_table.c.stid,
                func.min(price_table.c.date).label("first"),
                func.max(price_table.c.date).label("last")
            )
            
            #shorten list of price changes if possible
            if active_before and active_after:
                sq = sq.filter(sqlalchemy.or_(
                    price_table.c.date <= active_before,
                    price_table.c.date >= active_after
                ))
            elif active_before:
                sq = sq.filter(price_table.c.date <= active_before)
            elif active_after:
                sq = sq.filter(price_table.c.date >= active_after)
                
            #add subquery to actual query
            sq = sq.group_by(price_table.c.stid).subquery()
            query = query.join(sq, table.c.id == sq.c.stid)
            if active_before:
                query = query.filter(sq.c.first <= active_before)
            if active_after:
                query = query.filter(sq.c.last >= active_after)
        
        #create pandas dataframe
        return pd.read_sql(query.statement, self.connection).set_index("id")
        
    def find_price_history(self, stids, start=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc), end=pytz.utc.localize(datetime.utcnow())):
        """
        Create a pandas dataframe containing all price changes with the given
        properties.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations
        start -- the first update time to include in the price history (default
        datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc))
        end -- the last update time to include in the price history (default
        datetime.now())

        Return value:
        a pandas dataframe containing all information about the price changes
        from the database
        """
        #construct query
        table = self.meta.tables["gas_station_information_history"]
        query = self.session.query(table).filter(
            table.c.date >= start
        ).filter(
            table.c.date <= end
        ).filter(
            table.c.stid.in_(stids)
        ).order_by(
            table.c.date
        )
        
        #create pandas dataframe
        df = pd.read_sql(query.statement, self.connection)
        try:
            df["date"] = [pytz.utc.localize(t) for t in pd.to_datetime(df["date"], utc=True)]
        except:
            df["date"] = [t.astimezone(pytz.utc) for t in pd.to_datetime(df["date"], utc=True)]
        return df.set_index("date")
        
    def find_price_hourly_history(self, stid, start=datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc), end=pytz.utc.localize(datetime.utcnow()), fuel_type="diesel"):
        """
        Create a pandas series containing the hourly price at a given gas
        station.
        
        Keyword arguments:
        stid -- the id of a gas stations
        start -- the first update time to include in the price history (default
        datetime(2014, 7, 1, 0, 0, 0, 0, pytz.utc))
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
        current_price = self.find_prices([stid], time=start, fuel_types=[fuel_type]).iloc[0][fuel_type]

        #uncompress price data
        index_compressed = 0
        history_data = []
        index = pd.date_range(start=start, end=end, freq="1H", tz=pytz.utc)
        for date in index:
            if index_compressed < len(history_compressed) and history_compressed.index[index_compressed] <= date:
                current_price = history_compressed.iloc[index_compressed]
                index_compressed += 1
            history_data.append(current_price)
        history = pd.Series(history_data, index=index)

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
if __name__ == "__main__":
    db = Database()
    print(db.find_stations(place="Kassel"))
