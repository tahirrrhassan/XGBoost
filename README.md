# XGBoost on Firewall Dataset

This project demonstrates the application of the **XGBoost** machine learning algorithm for binary classification using the **Firewall dataset** from the UCI Machine Learning Repository. The objective is to predict whether network traffic is legitimate or malicious. The project includes data preprocessing, model training, performance evaluation, and visualizations using **Seaborn**.

## Overview
The assignment applies **XGBoost** to classify network traffic data from the **Firewall dataset** into two categories: legitimate and malicious. The project includes the following steps:
- Data preprocessing (handling missing values, encoding categorical variables, scaling features)
- Training the **XGBoost** model for binary classification
- Evaluating the model's performance using accuracy, precision, recall, and F1-score
- Visualizing the results, including the **Confusion Matrix** and **Feature Importance**, using **Seaborn**

## Project Structure
- **xgBoost.py**: Python script containing the implementation of the **XGBoost** model for firewall traffic classification.
- **firewall_logs/**: Directory containing the **Firewall dataset** from UCI.
- **xgBoost (21-MS-IEM-02).pdf**: Detailed assignment report (if required for submission).

## Prerequisites
To run the Python script, ensure the following libraries are installed:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `xgboost`
- `scikit-learn`

Install the required libraries using the following command:

```bash
pip install numpy pandas seaborn matplotlib xgboost scikit-learn
```

## Running the Code
1. **Dataset Placement**: Ensure the Firewall dataset is placed in the firewall_dataset/ folder.
2. **Execute Script**: Run the Python script using the following command:
   ```bash
   python xgboost.py
   ```
4. **Outputs**: The script will output the model’s accuracy, confusion matrix, and feature importance. These results will be displayed using Seaborn visualizations.

## Visualizations
Seaborn is used for the following visualizations:
- **Confusion Matrix**: Displays the number of correct and incorrect predictions, showing the accuracy of the model.
- **Feature Importance**: Visualizes the most significant features contributing to the prediction of whether network traffic is legitimate or malicious.

## Results Summary
- The XGBoost model achieves an accuracy of 90-95% on the test dataset, indicating strong performance.
- Key influential features:
  - Packet_Size
  - Flow_Bytes
  - Protocol
  - Source_Port

## Submission Notes
Ensure the following items are accessible in the shared Google Drive link:
1. Python script (`xgBoost.py`)
2. Assignment report (`xgBoost (21-MS-IEM-02).pdf`)
3. Titanic dataset folder (`firewall_logs/`)

