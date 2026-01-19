import sys
import os
import webbrowser
from threading import Timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from flask import Flask, render_template, request, jsonify
from src.inference.predict import predict_with_details

app = Flask(__name__, template_folder="web_assets")

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.get_json(silent=True)
        if req is None:
            raise ValueError("No JSON received")

        input_data = {}
        input_data["tenure"] = float(req.get("tenure", 0))
        input_data["MonthlyCharges"] = float(req.get("MonthlyCharges", 0))
        input_data["TotalCharges"] = float(req.get("TotalCharges", 0))

        prediction, probability = predict_with_details(input_data)

        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else 0.0
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "prediction": None,
            "probability": 0.0
        })

if __name__ == "__main__":
    # Automatically open browser after 1 second
    Timer(1, open_browser).start()
    print("🚀 Opening browser automatically...")
    app.run(debug=True, use_reloader=False)