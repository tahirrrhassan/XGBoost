# XGBoost
XGBoost model for binary classification on the Kaggle Titanic dataset. Includes data preprocessing, model training, performance evaluation, and visualizations using Seaborn.

## Overview
This project demonstrates the application of the XGBoost machine learning algorithm to predict survival on the Titanic dataset from Kaggle. The assignment includes data preprocessing, training the XGBoost model, evaluating its performance, and visualizing results using Seaborn.

## Project Structure
- **xgboost.py**: Python script containing the implementation of the XGBoost model.
- **xgBoost (21-MS-IEM-02).pdf**: Detailed assignment report.
- **titanic/**: Directory containing the Titanic dataset from Kaggle.

## Prerequisites
To run the Python script, the following libraries are required:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `xgboost`
- `scikit-learn`
- Kaggle API for dataset access

Install the required libraries using the following command:
```bash
pip install numpy pandas seaborn matplotlib xgboost scikit-learn kaggle
```

## Running the Code
1. **Dataset Placement**: Ensure the Titanic dataset is located in the `titanic/` folder.
2. **Execute Script**: Run the Python script using the following command:
   ```bash
   python xgboost.py
   ```
4. **Outputs**: The script will output the model's accuracy, confusion matrix, and feature importance.

## Visualizations
Seaborn is used for the following visualizations:
- **Confusion Matrix**: Provides an overview of the model's predictions.
- **Feature Importance**: Highlights the most influential features in the prediction process.

## Results Summary
- Achieved an accuracy of 80-85% on the test dataset.
- Key influential features:
  - Passenger Class (Pclass)
  - Gender (Sex)
  - Ticket Fare (Fare)
  - Age

## Submission Notes
Ensure the following items are accessible in the shared Google Drive link:
1. Python script (`xgboost.py`)
2. Assignment report (`xgBoost (21-MS-IEM-02).pdf`)
3. Titanic dataset folder (`titanic/`)

