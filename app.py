import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

# Create the Flask app
app = Flask(__name__)
CORS(app)
# Load models based on file availability
# Determine the correct model path depending on the environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "customer_churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

try:
    # Load the model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print("✅ Successfully loaded model")
    else:
        print(f"⚠️ Could not find {model_path}")
        
    # Load the scaler
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        print("✅ Successfully loaded scaler")
    else:
        print(f"⚠️ Could not find {scaler_path}")

    if model is None:
        raise Exception("Failed to load model from any source")
            
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    # We'll let the app start anyway, but predictions will fail

@app.route("/")
def home():
    return jsonify({"message": "Customer Churn Prediction API is running on Render"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")
        
        # Since we don't have label encoders, use direct mapping for country
        print("Using hardcoded country encoding")
        country_mapping = {"France": 0, "Spain": 1, "Germany": 2}
        encoded_country = country_mapping.get(data["Cus_contry"], -1)
        
        if encoded_country == -1:
            return jsonify({
                "error": f"Invalid country value: {data['Cus_contry']}, must be one of {list(country_mapping.keys())}"
            }), 400

        # Extract and prepare all feature values
        input_features = [
            float(data["Cus_report"]),
            float(data["Cus_age"]),
            float(data["Cus_memb"]),
            float(data["Cus_Active"]),
            float(data["Cus_money"]),
            encoded_country
        ]
        print(f"Features: {input_features}")

        # Make sure the input is properly shaped for the scaler and model
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale if scaler is available
        if scaler is not None:
            input_data = scaler.transform(input_array)
            print("Data scaled for prediction")
        else:
            input_data = input_array
            print("No scaler found, using raw features")
            
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get probability if the model supports it
        try:
            probability = model.predict_proba(input_data)[0][1]
            probability_str = f"{probability:.2f}"
        except:
            probability_str = "N/A"
            
        # Since we don't have label encoders, use direct mapping for churn prediction
        churn_prediction = "Yes" if prediction[0] == 1 else "No"

        print(f"Prediction: {churn_prediction}, Probability: {probability_str}")
        return jsonify({
            "prediction": churn_prediction,
            "probability": probability_str
        })

    except Exception as e:
        import traceback
        print(f"Server Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# This is important to keep for local development,
# but Render will use the app variable above
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)