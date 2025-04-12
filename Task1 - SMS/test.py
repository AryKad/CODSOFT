import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    filtered_words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]  # Apply stemming
    return ' '.join(stemmed_words)

model = joblib.load("spam_detector.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
best_threshold = joblib.load("best_threshold.pkl")

# Function to predict spam or ham using best threshold
def predict_message(message):
    cleaned_message = preprocess_text(message)  # Preprocess input
    message_tfidf = tfidf_vectorizer.transform([cleaned_message])  # Transform to TF-IDF
    spam_probability = model.predict_proba(message_tfidf)[:, 1][0]  # Get spam probability

    return "Spam" if spam_probability > best_threshold else "Ham"

# Test with user input
custom_message = input("Enter a message to check: ")
print(f"\nPredicted Class: {predict_message(custom_message)}")
