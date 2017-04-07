from flask import Flask, render_template

#create flask app
app = Flask(__name__)

#front page
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

#run flask
app.run(debug=True)
