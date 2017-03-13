from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy

class Database:
    """A class representing the connection to the gas prices database"""
    
    connection = None
    meta = None
    
    def __init__(self, dbconfig=None):
        """Connect to the database using a config object containing the database credentials.
        
        If no `configdb` argument is given we use the credentials from the file `db.ini`"""
        
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
    
    def find_stations(self, place=None):
        """Create a pandas dataframe containing all gas stations with the given properties.
        
        Keyword arguments:
        place -- the city of the gas stations (default None)"""
        #construct query
        table = self.meta.tables["gas_station"]
        query = table.select()
        if place:
            query = query.where(table.c.place == place)
        
        #create pandas dataframe
        return pd.read_sql(query, self.connection)
        
    def find_prices(self, stids=[], start=datetime.now() - timedelta(days=14), end=datetime.now()):
        """Create a pandas dataframe containing all price changes with the given properties.
        
        Keyword arguments:
        stids -- an iterable containing the ids of the gas stations (default [])
        start -- the first update time to include in the price history (default datetime.now() - timedelta(days=14))
        end -- the last update time to include in the price history (default datetime.now())"""
        #construct query
        table = self.meta.tables["gas_station_information_history"]
        query = table.select(sqlalchemy.and_(
            table.c.date >= start, table.c.date <= end, table.c.stid.in_(stids)
        ))
        
        #create pandas dataframe
        return pd.read_sql(query, self.connection)
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Close connection to the database"""
        self.connection.close()

#sample usage    
from ConfigParser import SafeConfigParser
if __name__ == "__main__":
    db = Database()
    print db.find_stations(place="Kassel").describe()
