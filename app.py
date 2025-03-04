import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# In place of print statements, use logger
try:
    # Load the scaler
    if os.path.exists(scaler_path):
        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("✅ Successfully loaded scaler")
    else:
        logger.warning(f"⚠️ Could not find {scaler_path}")
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Successfully loaded model")
    else:
        logger.warning(f"⚠️ Could not find {model_path}")

except Exception as e:
    logger.error(f"❌ Error during setup: {str(e)}")
    # Let the app start anyway, but predictions will fail

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        logger.error("Model not available. Check server logs.")
        return jsonify({"error": "Model not available. Check server logs."}), 500
        
    try:
        logger.info("📩 Raw request data: %s", request.data)
        if not request.data:
            return jsonify({"error": "Empty request body"}), 400
        
        data = request.get_json()
        
        if data is None:
            return jsonify({"error": "Invalid JSON format"}), 400

        logger.info("✅ Received JSON: %s", data)
        
        # Country encoding
        logger.info("Using hardcoded country encoding")
        country_mapping = {"France": 0, "Spain": 1, "Germany": 2}
        encoded_country = country_mapping.get(data["Cus_contry"], -1)
        
        if encoded_country == -1:
            return jsonify({
                "error": f"Invalid country value: {data['Cus_contry']}, must be one of {list(country_mapping.keys())}"
            }), 400

        # Prepare features
        input_features = [
            float(data["Cus_report"]),
            float(data["Cus_age"]),
            float(data["Cus_memb"]),
            float(data["Cus_Active"]),  
            float(data["Cus_money"]),
            encoded_country
        ]
        logger.info("Features: %s", input_features)

        # Transform input
        input_array = np.array(input_features).reshape(1, -1)
        
        if scaler is not None:
            input_data = scaler.transform(input_array)
            logger.info("Data scaled for prediction")
        else:
            input_data = input_array
            logger.info("No scaler found, using raw features")
            
        # Prediction
        prediction = model.predict(input_data)
        
        # Probability estimation
        try:
            probability = model.predict_proba(input_data)[0][1]
            probability_str = f"{probability:.2f}"
        except Exception as e:
            logger.error("Error obtaining prediction probability: %s", str(e))
            probability_str = "N/A"
        
        churn_prediction = "Yes" if prediction[0] == 1 else "No"

        logger.info("Prediction: %s, Probability: %s", churn_prediction, probability_str)
        return jsonify({
            "prediction": churn_prediction,
            "probability": probability_str,
            "model_type": "fallback"
        })

    except Exception as e:
        logger.error("Server Error: %s", str(e))
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500