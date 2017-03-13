import sqlalchemy

class Database:
    """A class representing the connection to the gas prices database"""
    
    connection = None
    meta = None
    
    def __init__(self, dbconfig):
        """Connect to the database using a config object containing the database credentials"""
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
        """Find the ids of all gas stations with the given properties"""
        #construct query
        table = self.meta.tables["gas_station"]
        query = table.select()
        if place:
            query = query.where(table.c.place == place)
        
        #iterate over results
        stations = []
        for station in self.connection.execute(query):
            stations.append(station["id"])
        return stations
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Close connection to the database"""
        self.connection.close()

#sample usage    
from ConfigParser import SafeConfigParser
if __name__ == "__main__":
    dbconfig = SafeConfigParser()
    dbconfig.read("db.ini")
    db = Database(dbconfig=dbconfig)
    print db.find_stations(place="Kassel")
