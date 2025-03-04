import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model = joblib.load("customer_churn_model.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running on Render"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        print(f"🔍 LabelEncoder for contry: {label_encoders['Cus_contry'].classes_}")

        if data["sex"] not in label_encoders["Cus_contry"].classes_:
            print(f"Invalid contry value received: {data['Cus_contry']}")
            return jsonify({
                "error": f"Invalid contry value: {data['Cus_contry']}, must be one of {label_encoders['Cus_contry'].classes_}"
            }), 400

        country_mapping = {"France": 0, "Spain": 1, "Germany": 2}
        encoded_country = country_mapping.get(data["Cus_contry"], -1)
        print(f"Encoded contry: {encoded_country}")

        features = [
            float(data["Cus_report"]),
            float(data["Cus_age"]),
            float(data["Cus_memb"]),
            float(data["Cus_Active"]),  
            float(data["Cus_money"]),
            encoded_country
        ]

        print(f"Features: {features}")

        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)
        probability  = label_encoders["churn"].inverse_transform([prediction[0]])[0]
        probability_str = f"{probability:.2f}"
        churn_prediction = "Yes" if prediction[0] == 1 else "No"
        print(f"Prediction: {churn_prediction}")
        return jsonify({
            "prediction": churn_prediction,
            "probability": probability_str,
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)