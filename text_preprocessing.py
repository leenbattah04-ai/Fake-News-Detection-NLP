# Fake News Detection - NLP Project

import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Download stopwords dataset (run once)
nltk.download('stopwords')

# Load dataset
news_dataset = pd.read_csv("train.csv")

# Display dataset shape and preview
print("Dataset shape:", news_dataset.shape)
print(news_dataset.head())

# Handle missing values
news_dataset = news_dataset.fillna('')

# Combine author and title into one text column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Initialize NLP tools
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [
        port_stem.stem(word)
        for word in content
        if word not in stop_words
    ]
    return ' '.join(content)

# Apply text preprocessing
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Split features and labels
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)