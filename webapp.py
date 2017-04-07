import json

from flask import Flask, render_template

from database import Database
from configparser import SafeConfigParser


#create flask app
app = Flask(__name__)

#front page
@app.route("/")
def index():
    return render_template("index.html")

#predictions page
@app.route('/api/predictions/<stid>')
def predictions(stid):
    config = SafeConfigParser()
    config.read("config.ini")
    return render_template("predictions.html",
        station=Database().find_stations(stids=[stid]).reset_index().iloc[0].to_dict(),
        apikey=config.get("gmaps", "apikey")
    )

#list all gas stations
@app.route("/api/stations")
def stations():
    data = Database().find_stations().reset_index()[["id", "name", "brand", "street", "place", "post_code"]]
    data_dict = data.to_dict(orient="records")
    return json.dumps(data_dict)

#run flask
app.run(host="0.0.0.0", debug=True)
