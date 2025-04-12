# ========== 📦 IMPORTS ==========
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# ========== 📥 SETUP ==========
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ========== 🧹 TEXT CLEANING ==========
def clean_plot(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# ========== 📂 LOAD FUNCTIONS ==========
def load_data(path, has_genre=True):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if has_genre and len(parts) == 4:
                movie_id, title, genre, plot = parts
                data.append({"id": int(movie_id), "title": title.strip(), "genre": genre.strip().lower(), "plot": plot.strip()})
            elif not has_genre and len(parts) == 3:
                movie_id, title, plot = parts
                data.append({"id": int(movie_id), "title": title.strip(), "plot": plot.strip()})
    return pd.DataFrame(data)

# ========== 📁 LOAD DATA ==========
train_path = "C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/train_data.txt"
test_path = "C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/test_data.txt"
test_solution_path = "C:/Users/Admin/Downloads/archive (5)/Genre Classification Dataset/test_data_solution.txt"

train_df = load_data(train_path, has_genre=True)
test_df = load_data(test_path, has_genre=False)
test_solution_df = load_data(test_solution_path, has_genre=True)

# ========== 🧽 CLEAN ==========
train_df["cleaned_plot"] = train_df["plot"].apply(clean_plot)
test_df["cleaned_plot"] = test_df["plot"].apply(clean_plot)
test_solution_df["cleaned_plot"] = test_solution_df["plot"].apply(clean_plot)

# ========== 🧠 S-BERT ENCODING ==========
print("🔄 Encoding with S-BERT...")
X_train = embedder.encode(train_df["cleaned_plot"].tolist(), show_progress_bar=True, batch_size=32)
X_test = embedder.encode(test_df["cleaned_plot"].tolist(), show_progress_bar=True, batch_size=32)

# ========== 🔠 LABEL ENCODING ==========
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["genre"])
y_test = label_encoder.transform(test_solution_df["genre"])

# ========== 🚀 TRAIN XGBOOST ==========
print("🚀 Training XGBoost...")
model = XGBClassifier(
    max_depth=8,
    n_estimators=150,
    learning_rate=0.1,
    objective='multi:softprob',
    eval_metric='mlogloss',
    verbosity=1
)
model.fit(X_train, y_train)

# ========== 🎯 PREDICTIONS ==========
y_pred = model.predict(X_test)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# ========== 📊 EVALUATION ==========
print("\n🎯 Classification Report (XGBoost + S-BERT on Test Data):")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
