from datetime import datetime, timedelta
import os

from ediblepickle import checkpoint
import pytz
import requests
from retrying import retry
from sklearn import base

from icalendar import Calendar
import pandas as pd


#create cache directory
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

@retry(stop_max_attempt_number=3)
@checkpoint(key=lambda args, kwargs: "calendar-" + ("-".join(map(str, args))), work_dir=cache_dir)
def download_holidays(year, county, calendar):
    """
    Download an ical calendar containing holiday data. This function retries in
    case of internet problems and uses caching of the calendars.
    
    Keyword arguments:
    year -- the year of the calendar to load
    county -- the county of Germany to load the data for in lower case with
    umlauts replaced by ae, oe, or ue
    calendar == the type of the calendar ("ferien" or "feiertage")
    
    Return value:
    the contents of the ical file 
    """
    url = "http://www.schulferien-deutschland.net/ical/%s-%s-%d.ics" % (calendar, county, year)
    return requests.get(url).text

def holidays(index, county, calendar_types = ["ferien", "feiertage"]):
    """
    Compute a pandas dataframe for a given time index that indicates whether
    there were public or school holidays that day.
    
    Keyword arguments:
    index -- a time series that should be used as the index of the dataframe
    county -- the county of Germany to load the data for in lower case with
    umlauts replaced by ae, oe, or ue
    calendar_types -- the columns to compute (default ["ferien", "feiertage"])
    
    Return value:
    the pandas dataframe as described above
    """
    #set all values to 0 at first, update later in case of holidays
    ret = pd.DataFrame({calendar_type: 0 for calendar_type in calendar_types}, index=index)
    for calendar_type in calendar_types:
        last_year = -1
        calendar = None
        for date in index:
            #get new calendar if the year changed
            if not last_year == date.year:
                calendar = Calendar.from_ical(download_holidays(date.year, county, calendar_type))
                last_year = date.year
            #update entries in the dataframe
            for t in calendar.walk("vevent"):
                if t['DTSTART'].dt <= date.date() and date.date() <= t['DTEND'].dt:
                    ret.loc[date][calendar_type] = 1
    return ret

@retry(stop_max_attempt_number=3)
@checkpoint(key=lambda args, kwargs: "zipcodes", work_dir=cache_dir)
def download_zipcodes():
    """
    Download a list of cities, their zipcodes and counties in Germany. This
    file is cached to reduce latency.
    
    Return value:
    a csv file
    """
    return requests.get("https://www.suche-postleitzahl.org/download_files/public/zuordnung_plz_ort.csv").text

def zipcode_to_county(zipcode):
    """
    Convert a German zipcode to the name of the corresponding county.
    
    Keyword arguments:
    zipcode -- the zipcode as a string
    
    Return value:
    the county at the given zipcode in lower case with umlauts replaced by
    ae, oe, and ue
    """
    csv = download_zipcodes().split("\n")
    columns = {}
    for i, column in enumerate(csv[0].split(",")):
        columns[column] = i
    for row in csv[1:]:
        try:
            split = row.split(",")
            if split[columns["plz"]] == zipcode:
                return split[columns["bundesland"]].replace("ü", "ue").lower()
        #in case of errors ignore that line
        except:
            pass
    return None

class HolidayTransformer(base.BaseEstimator, base.TransformerMixin):
    """
    A transformer that computes public and school holiday features in Germany.
    Features are given with values 0 or 1. The constructor needs to be given
    the zipcode of a place to compute the holidays for.
    """
    def __init__(self, zipcode):
        self.zipcode = zipcode
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        county = zipcode_to_county(self.zipcode)
        #What to do in case of lookup errors? We use a default value...
        return holidays(X.index, county if county else "berlin")

if __name__ == "__main__":
    #test calls for Stuttgart (in a county with umlaut)
    print(holidays(pd.date_range(
        start=datetime.utcnow(),
        end=datetime.utcnow() + timedelta(days=365),
        freq="1H",
        tz=pytz.utc
    ), zipcode_to_county("70173")))
