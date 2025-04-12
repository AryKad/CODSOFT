# ========== 📦 IMPORTS ==========
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ========== 📥 SETUP ==========
# nltk.download('stopwords')
# nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ========== 🔍 LOAD FUNCTIONS ==========
def load_train_data(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _, title, genre, plot = parts
                rows.append({
                    "title": title.strip(),
                    "genre": genre.strip().lower(),
                    "plot": plot.strip()
                })
    return pd.DataFrame(rows)

def load_test_data(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                _, title, plot = parts
                rows.append({
                    "title": title.strip(),
                    "plot": plot.strip()
                })
    return pd.DataFrame(rows)

def load_test_solution(file_path):
    genres = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                genres.append(parts[2].strip().lower())
    return genres

# ========== ✂️ TEXT CLEANING ==========
def clean_plot(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ========== 📁 LOAD DATA ==========
train_df = load_train_data("C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/train_data.txt")
test_df = load_test_data("C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/test_data.txt")
test_true_genres = load_test_solution("C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/test_data_solution.txt")

# ========== 🧹 CLEAN ==========
train_df["cleaned_plot"] = train_df["plot"].apply(clean_plot)
test_df["cleaned_plot"] = test_df["plot"].apply(clean_plot)

# ========== 🧠 TF-IDF + NAIVE BAYES ==========
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df["cleaned_plot"])
y_train = train_df["genre"]
X_test = vectorizer.transform(test_df["cleaned_plot"])

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== 📊 EVALUATION ==========
print("\n🎯 Classification Report (Multinomial Naive Bayes on Test Data):")
print(classification_report(test_true_genres, y_pred, zero_division=0))
