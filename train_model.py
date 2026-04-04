import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def scrub_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading dataset...")
df = pd.read_csv('UpdatedResumeDataSet.csv')
df['Cleaned'] = df['Resume'].apply(scrub_text)

print("Vectorising text...")
vectoriser = TfidfVectorizer(stop_words='english', max_features=1500)
X_train_vec = vectoriser.fit_transform(df['Cleaned'])

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, df['Category'])

print("Saving model and vectoriser...")
joblib.dump(model, 'resume_model.pkl')
joblib.dump(vectoriser, 'tfidf_vectoriser.pkl')
print("Done! You can now run the Flask app.")