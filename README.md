# Network Anomaly Detection Project

## Overview
This repository contains all the necessary files and resources for implementing and evaluating models for network anomaly detection. The project involves both supervised and unsupervised learning approaches, and the models are deployed as Flask applications for real-world usability.

---

## Repository Structure

### 1. `data`
- **`Network_anomaly_data.csv`**: The primary dataset containing various features related to network traffic, used for both training and testing the models.

### 2. `experiments`
- **`Possible Hypotheses to Test.ipynb`**: A Jupyter notebook listing and analyzing hypotheses that can be explored using the data.
- **`Supervised Learning Models and Evaluations.ipynb`**: A notebook showcasing supervised learning model implementations and their performance evaluations.
- **`Supervised Learning Models.ipynb`**: Detailed implementations of various supervised learning models.
- **`Unsupervised Learning Models and Evaluations.ipynb`**: A notebook focused on unsupervised learning models and their performance evaluations.
- **`Unsupervised Learning Models.ipynb`**: Detailed implementations of various unsupervised learning models.

### 3. `falsk_supervised_app`
- **`model`**: Directory containing pre-trained supervised learning models.
  - **`random_forest_model.joblib`**: A trained Random Forest model used for classification.
- **`app.py`**: Flask application script for deploying the supervised learning model as an API.
- **`requirements.txt`**: List of Python dependencies required to run the supervised learning Flask app.

### 4. `falsk_unsupervised_app`
- **`model`**: Directory containing pre-trained unsupervised learning models.
  - **`isolation_forest_model.joblib`**: A trained Isolation Forest model used for anomaly detection.
- **`app.py`**: Flask application script for deploying the unsupervised learning model as an API.
- **`requirements.txt`**: List of Python dependencies required to run the unsupervised learning Flask app.

---

## How to Use

### 1. Data Preparation
- Ensure that the dataset (`Network_anomaly_data.csv`) is present in the `data` folder.

### 2. Experimentation
- Use the Jupyter notebooks in the `experiments` folder to explore, train, and evaluate different models.

### 3. Model Deployment
- Navigate to the respective Flask application directories (`falsk_supervised_app` or `falsk_unsupervised_app`) for deploying the models.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
- Run the Flask app:
  ```bash
  python app.py
  ```

### 4. API Endpoints
- Supervised Learning App: Provides predictions based on the `random_forest_model.joblib`.
- Unsupervised Learning App: Detects anomalies using the `isolation_forest_model.joblib`.

---

## Requirements
- Python 3.8+
- Flask
- Joblib
- Additional dependencies listed in `requirements.txt` for each Flask app.

---

## Future Improvements
- Adding more advanced machine learning and deep learning models.
- Exploring additional datasets for enhanced generalization.
- Integrating a front-end interface for better user interaction.

---

## Contributors
- **Aniruddha Mukherjee**: Core developer and architect.

Feel free to contribute by submitting issues or pull requests!

