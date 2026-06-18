import pandas as pd
import numpy as np
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the NLP package downloads
nltk.download('stopwords')

# =========================================================
# UPDATE: LOAD AND MERGE OLD SMS & MODERN TELEGRAM DATASETS
# =========================================================

# 1. Load the original Kaggle SMS Dataset
df_sms = pd.read_csv('spam.csv', encoding='latin-1')
df_sms = df_sms[['v1', 'v2']]
df_sms.columns = ['label', 'message']

# 2. Load the new Telegram Dataset 
df_tele = pd.read_csv('telegram_spam.csv')
df_tele = df_tele[['text_type', 'text']]
df_tele.columns = ['label', 'message']

# 3. Standardize the labels to lowercase string format before merging
df_sms['label'] = df_sms['label'].astype(str).str.lower().str.strip()
df_tele['label'] = df_tele['label'].astype(str).str.lower().str.strip()

# 4. Merge them! Stacking them vertically on top of each other
df = pd.concat([df_sms, df_tele], ignore_index=True)

# 5. Map text targets to binary integers (0 = Ham, 1 = Spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 6. Drop any row that might have broken or missing values from the merge
df = df.dropna()

print(f"Datasets successfully merged! Total rows available for training: {len(df)}")

# =========================================================
# THE REST OF YOUR PIPELINE STAYS EXACTLY THE SAME
# =========================================================

# Define the text cleaning (NLP Preprocessing) pipeline
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # Normalize case
    text = ''.join([char for char in text if char not in string.punctuation]) # Strip punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Filter stopwords
    return text

# Apply preprocessing to build our clean text feature column
df['clean_message'] = df['message'].apply(clean_text)

# Separate features (X) from the target label (y)
X = df['clean_message']
y = df['label']

# Split into 80% Training data and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the mathematical vectorizer engine
vectorizer = TfidfVectorizer()

# Vectorization transformation pipeline
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and fit the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Run testing inference
y_pred = model.predict(X_test_vec)

# Calculate system performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Training Complete! Testing Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer to disk as hard files
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Saved model objects successfully to disk!")