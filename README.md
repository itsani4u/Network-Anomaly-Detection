# Network Anomaly Detection Project

## Overview
This repository contains all the necessary files and resources for implementing and evaluating models for network anomaly detection. The project involves both supervised and unsupervised learning approaches, and the models are deployed as Flask applications for real-world usability.

## Blog: https://medium.com/@itsani4u24/end-to-end-network-anomaly-detection-for-cybersecurity-a-comprehensive-guide-9148e40d8029

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

## Problem Statement
Network security is a critical concern, with anomalies potentially indicating cyber-attacks or system malfunctions. This project aims to detect these anomalies efficiently by leveraging machine learning models. 

## Target Metric
The primary goal is to achieve high accuracy and F1-scores for both supervised and unsupervised models, ensuring reliable anomaly detection.

## Steps Taken

### 1. Exploratory Data Analysis (EDA)
- Analyzed the dataset to identify patterns, trends, and missing values.
- Visualized distributions and correlations among features.

### 2. Hypothesis Testing
- Explored various hypotheses regarding the impact of features on anomaly detection.

### 3. Machine Learning Modeling
- Implemented supervised models: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Support Vector Machines (SVM), Neural Networks, and a Voting Classifier.
- Developed an unsupervised model: Isolation Forest for detecting anomalies without labeled data.
- Evaluated models using accuracy, precision, recall, and F1-scores.

## Insights and Recommendations
- **Best Model**: Random Forest achieved the highest accuracy of 99.95% based on cross-validation.
- **Insights**: The Voting Classifier also performed exceptionally well, combining the strengths of multiple models.
- Models like Logistic Regression and SVM showed limited effectiveness with accuracy around 53%, highlighting their unsuitability for this dataset.

## Final Scores
- **Voting Classifier**: Accuracy = 99.88%
- **Random Forest**: Accuracy = 99.95%
- **Bagging**: Accuracy = 99.91%
- **Gradient Boosting**: Accuracy = 99.73%
- **Neural Network**: Accuracy = 96.22%
- **Logistic Regression**: Accuracy = 53.46%
- **SVM**: Accuracy = 53.46%

### Classification Report Example (Random Forest):
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     20203
           1       1.00      1.00      1.00     17589

    accuracy                           1.00     37792
   macro avg       1.00      1.00      1.00     37792
weighted avg       1.00      1.00      1.00     37792
```


## Unsupervised experiments overview
This project demonstrates clustering and anomaly detection techniques using DBSCAN and Isolation Forest on a dataset processed with PCA. It includes k-distance plots for DBSCAN parameter tuning and visualizations for core, edge, and noise points.

The solution provides:
- **Dimensionality Reduction**: PCA for feature reduction while retaining 95% variance.
- **Clustering**: Using DBSCAN to identify clusters and outliers.
- **Anomaly Detection**: Employing Isolation Forest for robust anomaly identification.
- **Visualization**: Clear and detailed visualizations for clusters and anomalies.

## Key Features
1. **DBSCAN Optimization**:
   - K-distance plot to determine optimal `eps`.
   - Parameter tuning for `eps` and `min_samples` to achieve the best silhouette score.
   - Separate visualizations for noise, core, and edge points.

2. **Isolation Forest**:
   - Identifies anomalies based on feature distributions.
   - Works well for datasets with high-dimensional features after PCA.

3. **Visualizations**:
   - Clusters with distinct colors.
   - Noise points in red.
   - Silhouette scores for evaluating clustering performance.

---


## Rational for Choosing Isolation Forest
Isolation Forest is particularly suitable for anomaly detection due to the following reasons:

1. **Efficiency**:
   - Isolation Forest has a linear time complexity, making it faster for large datasets compared to techniques like Local Outlier Factor (LOF).

2. **Feature Independence**:
   - It performs well without requiring features to have specific distributions or relationships.

3. **Effective in High Dimensions**:
   - Unlike DBSCAN, which may struggle with high-dimensional data, Isolation Forest isolates anomalies more effectively after PCA.

4. **Robustness to Noise**:
   - It explicitly isolates anomalies, minimizing the impact of noisy data on the results.

---

## Steps to Run the Code
1. **Prepare the Environment**:
   - Install required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Input Dataset**:
   - Replace `your_dataset.csv` with your dataset.

3. **Run the Script**:
   - Execute the Python script to:
     - Perform PCA.
     - Optimize DBSCAN parameters.
     - Apply DBSCAN and Isolation Forest.
     - Visualize results.

4. **Parameter Tuning**:
   - Adjust `eps` and `min_samples` for DBSCAN.
   - Experiment with `contamination` in Isolation Forest.

---

## Visual Outputs
1. **K-Distance Graph**:
   - Helps determine optimal `eps` for DBSCAN.
2. **Cluster Visualization**:
   - Shows distinct clusters and noise points in red.
3. **Isolation Forest Anomalies**:
   - Highlights anomalies using the contamination parameter.



## Deployment Steps
1. **Model Serialization**:
   - Used `joblib` to save the trained models for deployment.

2. **Flask API Development**:
   - Built RESTful APIs for both supervised and unsupervised models.
   - APIs accept network traffic data and return predictions or anomaly scores.

3. **Testing**:
   - Validated API endpoints with sample data to ensure robustness.

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

