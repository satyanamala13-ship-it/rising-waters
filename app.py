from flask import Flask,render_template, request, redirect
import joblib
import numpy as np
app = Flask(__name__)
model=joblib.load("model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temp"])
    humidity = float(request.form["humidity"])
    cloud = float(request.form["cloud"])
    prediction=model.predict([[temp,humidity,cloud]])
    prediction=[1]
    if prediction[0]==1:
        return redirect("/chance")
    else:
        return redirect("/no-chance")
@app.route("/chance")
def chance():
    return render_template("chance.html")
@app.route("/no-chance")
def no_chance():
    return render_template("no_chance.html")
if __name__ == "__main__":
    app.run(debug=True)