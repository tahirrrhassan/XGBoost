# Import necessary libraries
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load the Titanic dataset
data = pd.read_csv('titanic/train.csv')

print(data.head())

data['Age'].fillna(data['Age'].median(), inplace=True)  # Fill missing ages with median
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode
data.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)  # Drop unnecessary columns

# Encode categorical variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])  # Male = 0, Female = 1
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])  # Encode Embarked (C=0, Q=1, S=2)

# Define features (X) and target (y)
X = data.drop(columns=['Survived'])
y = data['Survived']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix format for XGBoost
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',  # Logarithmic loss
    'max_depth': 5,  # Depth of each tree
    'eta': 0.1,  # Learning rate
}

# Train the XGBoost model
num_round = 100  # Number of boosting rounds
bst = xgb.train(params, train_data, num_round)

# Make predictions on the test data
y_pred = bst.predict(test_data)
y_pred = [1 if p > 0.5 else 0 for p in y_pred]  # Convert probabilities to binary class (0 or 1)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize the confusion matrix using Seaborn
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Did not Survive", "Survived"], yticklabels=["Did not Survive", "Survived"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance plot
xgb.plot_importance(bst)
plt.title("Feature Importance")
plt.show()
