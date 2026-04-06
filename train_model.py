import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def scrub_text(text):
    
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove special characters/emojis but keep alphanumeric
    text = re.sub(r'[^a-z0-9\s]', ' ', text) 
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading dataset...")
df = pd.read_csv('UpdatedResumeDataSet.csv')

print("Cleaning resumes...")
df['Cleaned'] = df['Resume'].apply(scrub_text)

print("Vectorising text with N-grams...")

vectoriser = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    max_features=3000,
    sublinear_tf=True,
    min_df = 5
)

X = vectoriser.fit_transform(df['Cleaned'])
y = df['Category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Advanced Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Quick Accuracy Check
y_pred = model.predict(X_test)
print(f"Training Complete! Validation Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("Saving updated model and vectoriser...")
joblib.dump(model, 'resume_model.pkl')
joblib.dump(vectoriser, 'tfidf_vectoriser.pkl')

print("Done!")