from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the Random Forest model
model = load("model/random_forest_model.joblib")

@app.route('/')
def home():
    return "Welcome to the Random Forest API!"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()

        # Convert input data to numpy array
        input_data = np.array(data['features']).reshape(1, -1)

        # Predict the class
        prediction = model.predict(input_data)
        # Predict the probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Prepare response
        response = {
            "prediction": int(prediction[0]),  # Convert class label to int
            "probabilities": {
                str(i): float(prob) for i, prob in enumerate(probabilities)
            },
            "message": "Prediction successful!"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
