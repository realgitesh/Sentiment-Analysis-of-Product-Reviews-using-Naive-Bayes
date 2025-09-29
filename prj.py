
# Sentiment Analysis of Product Reviews using Naive Bayes


import numpy as np
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys
import os
import re


# Function to install missing packages


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# List of required packages
packages = ["pandas", "matplotlib", "seaborn",
            "scikit-learn", "nltk", "wordcloud"]

# Install any missing packages
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)


# Import libraries


# Ensure NLTK resources are downloaded


def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            if res == 'stopwords':
                stopwords.words('english')
            else:
                WordNetLemmatizer().lemmatize("test")
        except LookupError:
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)


download_nltk_resources()

# Load dataset

try:
    df = pd.read_csv("reviews.csv")  # Ensure reviews.csv is in the same folder
except FileNotFoundError:
    raise FileNotFoundError(
        "reviews.csv not found. Please place it in the same folder as this script.")

print(df.head())
print("\nClass distribution:\n", df['sentiment'].value_counts())


# Text cleaning function (fixed for Windows)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)  # remove special chars
    text = text.lower()

    # Simple tokenizer (split on spaces) instead of nltk.word_tokenize
    tokens = text.split()

    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


df['clean_review'] = df['review'].apply(clean_text)


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Build Pipeline: TF-IDF + Naive Bayes

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
    ('clf', MultinomialNB(alpha=1.0))
])

# Train
pipeline.fit(X_train, y_train)


# Evaluate

y_pred = pipeline.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=pipeline.classes_,
            yticklabels=pipeline.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Word Clouds


def generate_wordcloud(text, title, colormap):
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap=colormap).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()


# Simple Bar Chart (per-class accuracy)

cm = metrics.confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
correct = cm.diagonal()
total = cm.sum(axis=1)
accuracy_per_class = correct / total

plt.figure(figsize=(6, 4))
bars = plt.bar(pipeline.classes_, accuracy_per_class,
               color=["red", "blue", "green"])
plt.ylim(0, 1)
plt.ylabel("Accuracy per Class")
plt.title("Model Performance by Sentiment Class")

# Show percentages on top of bars
for bar, acc in zip(bars, accuracy_per_class):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{acc:.0%}", ha="center", va="bottom")

plt.show()

# Positive Reviews
positive_text = " ".join(df[df['sentiment'] == 'positive']['clean_review'])
generate_wordcloud(positive_text, "Word Cloud - Positive Reviews", "Greens")

# Negative Reviews
negative_text = " ".join(df[df['sentiment'] == 'negative']['clean_review'])
generate_wordcloud(negative_text, "Word Cloud - Negative Reviews", "Reds")

# Neutral Reviews
neutral_text = " ".join(df[df['sentiment'] == 'neutral']['clean_review'])
generate_wordcloud(neutral_text, "Word Cloud - Neutral Reviews", "Blues")


# Test on new examples

examples = [
    "The camera quality is fantastic and battery lasts long!",
    "Terrible product. It broke in 2 days.",
    "Delivery was okay, nothing special."
]
cleaned = [clean_text(x) for x in examples]
print("Predictions:", pipeline.predict(cleaned))
