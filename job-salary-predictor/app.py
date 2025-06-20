from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load the trained model
model = joblib.load("model/salary_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        experience = float(request.form["experience"])
        python = int(request.form.get("python", 0))
        excel = int(request.form.get("excel", 0))
        sql = int(request.form.get("sql", 0))

        features = np.array([[experience, python, excel, sql]])
        prediction = model.predict(features)
        salary = round(prediction[0])

        return render_template("index.html", salary=salary)
    except Exception as e:
        return render_template("index.html", salary="❌ Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)

