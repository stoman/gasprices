import json

from flask import Flask, render_template

from database import Database


#create flask app
app = Flask(__name__)

#front page
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

#list all gas stations
@app.route("/api/stations")
def stations():
    data = Database().find_stations().reset_index()[["id", "name", "brand", "street", "place", "post_code"]]
    data_dict = data.to_dict(orient="records")
    return json.dumps(data_dict)

#run flask
app.run(host="0.0.0.0", debug=True)
