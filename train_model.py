import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Dataset को load करना
print("Dataset load हो रहा है...")

data = pd.read_csv("dataset.csv")

print("Dataset successfully load हो गया!")
print("Total messages:", len(data))


# Data को अलग करना
X = data["message"]
y = data["label"]


# Spam = 1
# Ham = 0

y = y.map({
    "spam": 1,
    "ham": 0
})


# Training और Testing data बनाना
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Machine Learning Pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# Model को train करना
print("\nModel training शुरू हो रही है...")

model.fit(X_train, y_train)

print("Model training complete!")


# Model को test करना
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)


# Model को save करना
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel spam_model.pkl के नाम से save हो गया!")