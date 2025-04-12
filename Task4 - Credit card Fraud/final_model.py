import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# Load the saved Random Forest model
model = joblib.load("best_fraud_rf_model.pkl")

# Load the test data
test_df = pd.read_csv("C:/Users/Admin/Downloads/archive (3)/fraudTest.csv")

# Preprocessing function (same as training)
def preprocess(df):
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df.drop(columns=['trans_date_trans_time', 'unix_time', 'dob', 'merchant', 'trans_num',
                     'first', 'last', 'street', 'city', 'state', 'job'], inplace=True)
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    df = pd.get_dummies(df, columns=['category'], drop_first=True)
    return df

# Preprocess the test set
test_df_processed = preprocess(test_df)
X_test = test_df_processed.drop(columns='is_fraud')
y_test = test_df_processed['is_fraud']

# Predict using the loaded model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("🎯 Classification Report (Loaded RF Model on Test Data):")
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_score:.4f}")
