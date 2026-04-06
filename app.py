import os
import re
import joblib
import PyPDF2
import docx
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved ML components
try:
    model = joblib.load('resume_model.pkl')
    vectoriser = joblib.load('tfidf_vectoriser.pkl')
except:
    print("⚠️ Warning: Model files not found. Run train_model.py first!")

def scrub_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                content = page.extract_text()
                if content: text += content
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    return text

def is_valid_resume(text):
    """Checks if the document contains common resume keywords."""
    # List of words almost guaranteed to be in any resume
    keywords = ['experience', 'education', 'skills', 'projects', 'resume', 
                'profile', 'summary', 'university', 'college', 'degree', 'work']
    
    text_lower = text.lower()
    
    # Count how many of these keywords appear in the extracted text
    match_count = sum(1 for word in keywords if word in text_lower)
    
    # If it has at least 3 of these common sections, we accept it
    return match_count >= 3

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None # Add an error variable to send to HTML

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            
            # 1. Extract text
            raw_text = extract_text_from_file(path)
            
            # 2. VALIDATE THE DOCUMENT
            if not is_valid_resume(raw_text):
                os.remove(path) # Clean up the bad file
                error = "❌ This doesn't look like a valid resume. Please upload a real resume."
                return render_template('index.html', error=error)

            # 3. Clean
            cleaned = scrub_text(raw_text)
            
            # --- NEW DEBUGGING CODE ---
            print("\n--- DEBUG INFO ---")
            print(f"Text the model saw: {cleaned[:300]}...") 
            
            vec = vectoriser.transform([cleaned])
            prediction = model.predict(vec)[0]
            
            # Get the top 3 guesses and their percentages
            probabilities = model.predict_proba(vec)[0]
            classes = model.classes_
            
            # Sort to find the highest percentages
            top_3_indices = probabilities.argsort()[-3:][::-1]
            print("\nTop 3 Predictions:")
            for i in top_3_indices:
                print(f"{classes[i]}: {probabilities[i]*100:.2f}%")
            print("------------------\n")
            # --------------------------
            
            # Clean up: delete file after processing
            os.remove(path)
            
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)