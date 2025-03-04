import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from sklearn.ensemble import RandomForestClassifier  # or another algorithm you prefer

# Create the Flask app
app = Flask(__name__)
CORS(app)

# Initialize variables
model = None
scaler = None

# Determine the correct model path depending on the environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

try:
    # Load the scaler - this seems to be working
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Successfully loaded scaler")
    else:
        print(f"⚠️ Could not find {scaler_path}")
        
    # Since we can't load the pickled model, let's create a fallback model
    print("Creating a fallback model since the original model couldn't be loaded")
    # This is a simple placeholder model that will return predictions based on rules
    # Replace this with a proper trained model in production
    
    class FallbackModel:
        def __init__(self):
            print("Fallback model initialized")
            
        def predict(self, X):
            # Simple rule-based predictions
            # For demonstration: predict churn if credit score < 600 or balance < 1000
            predictions = []
            for sample in X:
                # Assuming features:
                # [Cus_report, Cus_age, Cus_memb, Cus_Active, Cus_money, encoded_country]
                credit_score = sample[0]  # Cus_report
                balance = sample[4]       # Cus_money
                
                if credit_score < 600 or balance < 1000:
                    predictions.append(1)  # Churn
                else:
                    predictions.append(0)  # No churn
            return np.array(predictions)
            
        def predict_proba(self, X):
            # Return pseudo-probabilities
            raw_preds = self.predict(X)
            # Create a 2D array where each row has probabilities for [no-churn, churn]
            probas = []
            for p in raw_preds:
                if p == 1:
                    probas.append([0.2, 0.8])  # Arbitrary values 
                else:
                    probas.append([0.7, 0.3])
            return np.array(probas)
    
    # Use the fallback model
    model = FallbackModel()
    print("✅ Fallback model loaded and ready")

except Exception as e:
    print(f"❌ Error during setup: {str(e)}")
    # We'll let the app start anyway, but predictions will fail

@app.route("/")
def home():
    return jsonify({"message": "Customer Churn Prediction API is running on Render", 
                    "status": "using fallback model"})

@app.route("/predict", methods=["POST"])
def predict():
    
    if model is None:
        return jsonify({"error": "Model not available. Check server logs."}), 500
        
    try:
        print("📩 Raw request data:", request.data)  # Print raw data for debugging

        if not request.data:  # Check if the request body is empty
            return jsonify({"error": "Empty request body"}), 400
        
        data = request.get_json()
        
        if data is None:  # get_json() returns None if JSON is invalid
            return jsonify({"error": "Invalid JSON format"}), 400

        print(f"✅ Received JSON: {data}")
        
        # Direct mapping for country
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
            
        # Map prediction to human-readable output
        churn_prediction = "Yes" if prediction[0] == 1 else "No"

        print(f"Prediction: {churn_prediction}, Probability: {probability_str}")
        return jsonify({
            "prediction": churn_prediction,
            "probability": probability_str,
            "model_type": "fallback"  # Indicate we're using the fallback model
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