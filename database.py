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
        Connect to the database using a config object containing the database credentials.
        
        If no `configdb` argument is given we use the credentials from the file `db.ini`

        Keyword arguments:
        configdb -- a `SafeConfigParser` containing database credentials (default None)
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
            ), client_encoding='utf8'
        )
        self.meta = sqlalchemy.MetaData(bind=self.connection, reflect=True)
        Session = sessionmaker(bind=self.connection)
        self.session = Session()
    
    def find_stations(self, place=None):
        """
        Create a pandas dataframe containing all gas stations with the given properties.
        
        Keyword arguments:
        place -- the city of the gas stations (default None)
        """
        #construct query
        table = self.meta.tables["gas_station"]
        query = self.session.query(table)
        if place:
            query = query.filter(table.c.place == place)
        
        #create pandas dataframe
        return pd.read_sql(query.statement, self.connection)
        
    def find_price_history(self, stids, start=datetime.now(pytz.utc) - timedelta(days=14), end=datetime.now(pytz.utc)):
        """
        Create a pandas dataframe containing all price changes with the given properties.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations
        start -- the first update time to include in the price history (default datetime.now(pytz.utc) - timedelta(days=14))
        end -- the last update time to include in the price history (default datetime.now(pytz.utc))
        """
        #construct query
        table = self.meta.tables["gas_station_information_history"]
        query = self.session.query(table).filter(sqlalchemy.and_(
            table.c.date >= start, table.c.date <= end, table.c.stid.in_(stids)
        ))
        
        #create pandas dataframe
        return pd.read_sql(query.statement, self.connection)
        
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close connection to the database
        """
        self.connection.close()
        
    def import_database(self, url="https://creativecommons.tankerkoenig.de/history/history.dump.gz"):
        """
        Imports a gzipped database file from an external server. Make sure you trust this server!
        
        Keyword arguments:
        url -- the url of the file to load (default "https://creativecommons.tankerkoenig.de/history/history.dump.gz")
        """
        response = urllib.request.urlopen(url)
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        self.connection.execute(decompressed_file.read())
        
    def create_indices(self):
        """
        Create some indices on the database for faster queries
        """
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_history_date_stid ON gas_station_information_history USING btree(date, stid);")
        
    def find_prices(self, stids, time=datetime.now(), fuel_types=["diesel", "e5", "e10"]):
        """
        Create a pandas dataframe containing the current prices at the given gas stations and the given time.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations
        time -- the time to find the gas prices (default datetime.now())
        fuel_types -- a list of fuel types to return (default ["diesel", "e5", "e10"])
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
    print(db.find_stations(place="Kassel").describe())
