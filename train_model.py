import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample dataset (you can expand later)
data = {
    "text": [
        "system attack detected",
        "malware found in file",
        "phishing attempt",
        "hello how are you",
        "normal login success",
        "user accessed dashboard"
    ],
    "label": [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = MultinomialNB()
model.fit(X, df["label"])

# Save model + vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")