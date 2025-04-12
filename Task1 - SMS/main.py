import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

df = pd.read_csv("C:/Users/Admin/Downloads/archive (2)/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
# Check for missing values
# print(df.isnull().sum())

# Check for duplicate rows
# print(f"Duplicates found: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates()

# Check new dataset size
# print(f"Dataset size after removing duplicates: {df.shape}")

# Plot spam vs ham distribution
# df['label'].value_counts().plot(kind='bar', color=['blue', 'red'])
# plt.title("Spam vs Ham Distribution")
# plt.xlabel("Category")
# plt.ylabel("Count")
# plt.xticks(rotation=0)
# plt.show()

# Download stopwords if not already downloaded
# nltk.download('stopwords')
# nltk.download('punkt_tab')  # Download missing tokenizer resource
# nltk.download('punkt')  # Ensure punkt is also available


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join words back into a string
    return ' '.join(stemmed_words)


df['cleaned_message'] = df['message'].apply(preprocess_text)

# View a sample
print(df[['message', 'cleaned_message']].head(10))

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features to 5000

# Apply TF-IDF transformation
X = tfidf_vectorizer.fit_transform(df['cleaned_message'])  # Convert text to TF-IDF matrix
y = df['label']  # Target variable (spam or not)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)


# Predict on test data
y_proba = model.predict_proba(X_test)[:, 1]  # Get probability scores


# y_pred = (y_proba > 0.2).astype(int)  # Change threshold to 0.4
# Convert predictions (0,1) back to original labels ('ham', 'spam')


# Convert y_test to numerical labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)  # 'ham' -> 0, 'spam' -> 1
precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_proba)

best_threshold = 0.5  # Default
best_f1 = 0

for threshold in thresholds:
    y_pred_temp = (y_proba > threshold).astype(int)
    f1 = f1_score(y_test_encoded, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal Threshold: {best_threshold}, Best F1 Score: {best_f1}")

# Use the best threshold
y_pred = (y_proba > best_threshold).astype(int)
y_pred_labels = ['ham' if pred == 0 else 'spam' for pred in y_pred]
y_pred_encoded = label_encoder.transform(y_pred_labels)
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_encoded))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_encoded))

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test_encoded, y_proba)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Naïve Bayes')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()
