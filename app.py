import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the trained model and scaler
model = joblib.load("customer_churn_model.pkl")  # Now it's a VotingClassifier
scaler = joblib.load("scaler.pkl")

# Label encoding for 'Geography'
geography_mapping = {"France": 0, "Spain": 1, "Germany": 2}

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        # Validate 'Geography'
        if data["Geography"] not in geography_mapping:
            return jsonify({
                "error": f"Invalid Geography value: {data['Geography']}, must be one of {list(geography_mapping.keys())}"
            }), 400

        encoded_geography = geography_mapping[data["Geography"]]
        print(f"Encoded Geography: {encoded_geography}")

        # Prepare input features
        features = [
            float(data["Complain"]),
            float(data["Age"]),
            float(data["IsActiveMember"]),
            float(data["NumOfProducts"]),
            float(data["Balance"]),
            encoded_geography
        ]

        features_array = np.array([features]).reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        # Prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]  # Probability of churn

        # Convert results
        churn_prediction = "Yes" if prediction == 1 else "No"
        probability_str = f"{probability:.2f}"

        print(f"Prediction: {churn_prediction}, Probability: {probability_str}")

        return jsonify({
            "prediction": churn_prediction,
            "probability": probability_str
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
