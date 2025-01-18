from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load("model/isolation_forest_model.joblib")

@app.route('/')
def home():
    return "Welcome to the Network Anomaly Using Unsupervised Algorithm!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        # Convert data to numpy array
        input_data = np.array(data['features']).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(input_data)
        response = {
            "prediction": int(prediction[0]),  # Convert to int for JSON serialization
            "message": "Anomaly" if prediction[0] == -1 else "Normal"
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
