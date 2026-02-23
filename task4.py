import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'text': [
        "Win a free lottery ticket",
        "Call now to claim prize",
        "Free entry in contest",
        "Congratulations you won money",
        "Meeting at 10 AM tomorrow",
        "Project discussion today",
        "Submit assignment by tonight",
        "Let's have lunch tomorrow"
    ],
    'label': [
        "spam",
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham",
        "ham"
    ]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
sample = ["Free prize waiting for you"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)

print("Prediction:", prediction[0])
