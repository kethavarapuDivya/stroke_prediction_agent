from flask import Flask, render_template, request
import joblib
import numpy as np
import threading
import webbrowser

app = Flask(__name__)

# Load trained model
model = joblib.load("stroke_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        gender = int(request.form["gender"])
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        ever_married = int(request.form["ever_married"])
        work_type = int(request.form["work_type"])
        residence_type = int(request.form["Residence_type"])
        avg_glucose_level = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking_status = int(request.form["smoking_status"])

        # Feature order must match the training data
        features = np.array([[
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            avg_glucose_level,
            bmi,
            smoking_status
        ]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Stroke"
        else:
            result = "✅ Low Risk of Stroke"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {e}")


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=5001, debug=True)
