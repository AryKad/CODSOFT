# ========== 📦 IMPORTS ==========
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========== 📥 SETUP ==========
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ========== 🔍 STEP 1: Load and Parse the Dataset ==========
def load_train_data(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                movie_id, title, genre, plot = parts
                rows.append({
                    "id": int(movie_id),
                    "title": title.strip(),
                    "genre": genre.strip().lower(),
                    "plot": plot.strip()
                })
    return pd.DataFrame(rows)

train_df = load_train_data("C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/train_data.txt")

# ========== ✂️ STEP 2: Text Preprocessing ==========
def clean_plot(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

train_df["cleaned_plot"] = train_df["plot"].apply(clean_plot)

# ========== 📊 STEP 3: TF-IDF Vectorization ==========
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X = vectorizer.fit_transform(train_df["cleaned_plot"])
y = train_df["genre"]

# ========== 🔀 STEP 4: Train-Test Split ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== 🌲 STEP 5: Train Random Forest ==========
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ========== 📈 STEP 6: Evaluate ==========
y_pred = model.predict(X_val)
print("\n🎯 Classification Report (Random Forest):")
print(classification_report(y_val, y_pred, zero_division=0))
