from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ✅ Load dataset
df = pd.read_csv("spam.csv")

# ✅ Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ✅ Clean messages
df['message'] = df['message'].astype(str).str.lower()

# ✅ Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# ✅ Store history
history = []

# ✅ Route
@app.route('/', methods=['GET', 'POST'])
def home():
    global history
    result = ""

    if request.method == 'POST':
        message = request.form['message'].lower()

        pred = model.predict(vectorizer.transform([message]))

        if pred[0] == 1:
            result = "Spam ❌"
        else:
            result = "Not Spam ✅"

        history.append(f"{message} → {result}")

    return render_template('index.html', result=result, history=history)


# ✅ RUN (works for both local + deployment)
if __name__ == "__main__":
    app.run()