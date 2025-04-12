import numpy as np
import pandas as pd
from _pickle import load
from sklearn import feature_extraction

# Load the trained model
filename = 'finalized_model.sav'
bayes = load(open(filename, 'rb'))

# Load the dataset to get the same TF-IDF vectorizer
data = pd.read_csv("C:/Users/Admin/Downloads/archive (2)/spam.csv", encoding='latin-1')

# Initialize the same TF-IDF vectorizer
f = feature_extraction.text.TfidfVectorizer(stop_words='english')
X = f.fit_transform(data["v2"])  # Fit on the training data

# Function to check new messages
def predict_message(msg):
    msg_transformed = f.transform([msg])  # Transform new message
    prediction = bayes.predict(msg_transformed)[0]  # Predict
    return "Spam" if prediction == 1 else "Ham"

# Example: Check if a message is spam or ham
new_message = input("Enter a message to check: ")
result = predict_message(new_message)
print(f"Message: {new_message}\nPrediction: {result}")
