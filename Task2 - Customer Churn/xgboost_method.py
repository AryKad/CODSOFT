import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# Drop unnecessary columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Encode categorical variables
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)  # One-hot encoding

# Split data into features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning for XGBoost
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]  # Addressing class imbalance
}

random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, tree_method='hist'),
    param_distributions=param_dist,
    n_iter=5,  # Reduced for faster tuning
    cv=3,
    scoring='accuracy',
    n_jobs=1,  # Single-threaded to avoid system overload
    verbose=2,  # Logs each step for debugging
    random_state=42
)
random_search.fit(X_train, y_train)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Model Evaluation
print("Best Parameters:", random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
