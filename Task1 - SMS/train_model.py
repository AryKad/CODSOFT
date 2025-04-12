import pandas as pd
import string
import joblib  # For saving & loading the model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:/Users/Admin/Downloads/archive (2)/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

# Remove duplicates
df = df.drop_duplicates()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    filtered_words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]  # Apply stemming
    return ' '.join(stemmed_words)

df['cleaned_message'] = df['message'].apply(preprocess_text)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'ham' -> 0, 'spam' -> 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Get probability scores for test data
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of being spam

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
best_threshold = 0.34734342088402137  # Use the threshold that gives best F1-score

# Save model & vectorizer
joblib.dump(model, "spam_detector.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(best_threshold, "best_threshold.pkl")
print("✅ Model, vectorizer, and threshold saved successfully!")

# Load everything for real-time prediction
model = joblib.load("spam_detector.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
best_threshold = joblib.load("best_threshold.pkl")

# Function to predict spam or ham using best threshold
def predict_message(message):
    cleaned_message = preprocess_text(message)  # Preprocess input
    message_tfidf = tfidf_vectorizer.transform([cleaned_message])  # Transform to TF-IDF
    spam_probability = model.predict_proba(message_tfidf)[:, 1][0]  # Get spam probability
    print(spam_probability)
    print(best_threshold)
    return "Spam" if spam_probability > best_threshold else "Ham"

# Test with user input
custom_message = input("Enter a message to check: ")
print(f"\nPredicted Class: {predict_message(custom_message)}")
