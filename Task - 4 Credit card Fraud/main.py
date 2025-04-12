import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv("C:/Users/Admin/Downloads/archive (3)/fraudTrain.csv")
test_df = pd.read_csv("C:/Users/Admin/Downloads/archive (3)/fraudTest.csv")

def preprocess(df):
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df.drop(columns=['trans_date_trans_time', 'unix_time', 'dob', 'merchant', 'trans_num', 'first', 'last', 'street', 'city', 'state', 'job'], inplace=True)
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    df = pd.get_dummies(df, columns=['category'], drop_first=True)
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

X_train = train_df.drop(columns='is_fraud')
y_train = train_df['is_fraud']
X_test = test_df.drop(columns='is_fraud')
y_test = test_df['is_fraud']

def plot_conf_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n🎯 Classification Report ({name}):")
    print(classification_report(y_test, y_pred))
    plot_conf_matrix(name, y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("🔄 Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("✅ SMOTE applied. Class distribution after resampling:")
print(pd.Series(y_train_sm).value_counts())

print("\n🚀 Training Logistic Regression...")
log_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
log_model.fit(X_train_sm, y_train_sm)
evaluate_model("Logistic Regression", log_model, X_test, y_test)

print("\n🚀 Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_sm, y_train_sm)
evaluate_model("Decision Tree", dt_model, X_test, y_test)

print("\n🚀 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_sm, y_train_sm)
evaluate_model("Random Forest", rf_model, X_test, y_test)

joblib.dump(rf_model, "best_fraud_rf_model.pkl")
