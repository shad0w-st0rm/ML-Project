# Import necessary libraries
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import gensim
import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Instantiate stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Load pre-trained Word2Vec model
word2vec = gensim.models.KeyedVectors.load("word2vec-google-news-300.kv")

# Define the cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove usernames
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return ' '.join(words)

# Vectorize a sentence by averaging its word vectors
def vectorize_sentence(text):
    words = text.split()
    avg_vector = np.zeros(word2vec.vector_size)
    valid_word_count = 0
    for word in words:
        if word in word2vec.key_to_index:
            avg_vector += word2vec[word]
            valid_word_count += 1
    if valid_word_count > 0:
        avg_vector /= valid_word_count
    return avg_vector

# Binary LSTM model
def binary_lstm_model(length):
    model_lstm = Sequential()
    model_lstm.add(LSTM(128, input_shape=(length, word2vec.vector_size), return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model_lstm

# Multiclass LSTM model
def multiclass_lstm_model(length, n_classes):
    model_lstm = Sequential()
    model_lstm.add(LSTM(128, input_shape=(length, word2vec.vector_size), return_sequences=True))
    model_lstm.add(LSTM(64, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(n_classes, activation='softmax'))
    model_lstm.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_lstm

# Process dataset, train models, and print results
def process_and_train(file_path, is_binary):
    X, y = [], []

    # Load and preprocess data
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            # Skip rows that don't have exactly 2 columns
            if len(line) != 2:
                print(f"Skipping malformed row {i} in {file_path}: {line}")
                continue

            review, label = line[0].strip(), line[1].strip()  # Strip spaces

            # Clean and vectorize the review
            cleaned_review = clean_text(review)
            avg_vector = vectorize_sentence(cleaned_review)

            try:
                # Handle binary or multi-class labels
                if is_binary:
                    label = 1 if label.lower() == "positive" else 0
                else:
                    label = int(label)  # Convert label to integer for multi-class
            except ValueError:
                print(f"Skipping invalid label in row {i}: {label}")
                continue

            X.append(avg_vector)
            y.append(label)

    if not X or not y:
        raise ValueError(f"No valid data found in {file_path}")

    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate traditional models
    results = {}

    # Logistic Regression
    model_lr = LogisticRegression(max_iter=250)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_valid)
    results["Logistic Regression"] = accuracy_score(y_valid, y_pred_lr)

    # Linear SVM
    model_svm = LinearSVC(max_iter=1000)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_valid)
    results["Linear SVM"] = accuracy_score(y_valid, y_pred_svm)

    # Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_valid)
    results["Random Forest"] = accuracy_score(y_valid, y_pred_rf)

    # Train and evaluate LSTM model
    if is_binary:
        max_len = 1  # Single vector per sentence
        y_train_lstm = np.array(y_train)
        y_valid_lstm = np.array(y_valid)
        padded_train = np.expand_dims(X_train, axis=1)
        padded_valid = np.expand_dims(X_valid, axis=1)

        model_lstm = binary_lstm_model(max_len)
        model_lstm.fit(padded_train, y_train_lstm, epochs=10, batch_size=128)
        y_pred_lstm = model_lstm.predict(padded_valid)
        y_pred_lstm = np.round(y_pred_lstm).flatten()
        print(f"LSTM Accuracy: {accuracy_score(y_valid_lstm, y_pred_lstm):.4f}")

    else:
        num_classes = len(set(y))
        y_train_lstm = to_categorical(y_train, num_classes=num_classes)
        y_valid_lstm = to_categorical(y_valid, num_classes=num_classes)
        padded_train = np.expand_dims(X_train, axis=1)
        padded_valid = np.expand_dims(X_valid, axis=1)

        model_lstm = multiclass_lstm_model(1, num_classes)
        model_lstm.fit(padded_train, y_train_lstm, epochs=10, batch_size=128)
        y_pred_lstm = model_lstm.predict(padded_valid)
        y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
        print(f"LSTM Accuracy: {accuracy_score(y_valid, y_pred_lstm):.4f}")

    # Print accuracy of traditional models
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")



# Call the process_and_train function for different datasets
dataset_files = [("financial_dataset.csv", False), ("hotel_dataset.csv", False), ("movie_dataset.csv", True)]
for file_path, is_binary in dataset_files:
    print(f"\nProcessing dataset: {file_path}")
    process_and_train(file_path, is_binary)
