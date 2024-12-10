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


# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Instantiate lemmatizer
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
    words = text.split()    # Split the words by spaces
    sentence_vector = []
    for word in words:
        if word in word2vec.key_to_index:   # If the word in the word2vec library
            sentence_vector.append(word2vec[word])  # then add the word2vec weights to the list
    return sentence_vector

def average_sentence(sentence_vector):  # this will convert a sentence (list of word vectors) into a single averaged vector
    avg_vector = np.zeros(word2vec.vector_size)
    for word in sentence_vector:
        avg_vector += word
    if len(sentence_vector) > 0: avg_vector /= len(sentence_vector)
    return avg_vector


# Binary LSTM model
def binary_lstm_model(length):
    model_lstm = Sequential()
    model_lstm.add(LSTM(128, input_shape=(length, word2vec.vector_size), return_sequences=False))    # input_shape is number of words in a sample by word vector size
    model_lstm.add(Dropout(0.2))   # Drop 20% of training samples (set to 0)
    model_lstm.add(Dense(1, activation='sigmoid'))  # Use sigmoid because binary classification
    model_lstm.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model_lstm

# Multiclass LSTM model
def multiclass_lstm_model(length, n_classes):
    model_lstm = Sequential()   # Create a sequential model
    model_lstm.add(LSTM(128, input_shape=(length, word2vec.vector_size), return_sequences=True))    # input_shape is number of words in a sample by word vector size
    model_lstm.add(LSTM(64, return_sequences=False))
    model_lstm.add(Dropout(0.2))    # Drop 20% of training samples (set to 0)
    model_lstm.add(Dense(n_classes, activation='softmax'))  # Use softmax because multiclass
    model_lstm.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy']) # categorical cross entropy (because classes were one hot encoded earlier, otherwise sparse crossentropy also works)
    return model_lstm

# Process dataset, train models, and print results
def process_and_train(file_path, is_binary, offset_output, lstm_params):
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
            sentence_vector = vectorize_sentence(cleaned_review)

            try:
                # Handle binary or multi-class labels
                if is_binary:
                    label = 1 if label.lower() == "positive" else 0
                else:
                    label = int(label) - offset_output  # Convert label to integer for multi-class and offset the output so it starts at 0
            except ValueError:
                print(f"Skipping invalid label in row {i}: {label}")
                continue

            X.append(sentence_vector)
            y.append(label)

    if not X or not y:
        raise ValueError(f"No valid data found in {file_path}")

    sentence_vectors = X
    X = [average_sentence(sentence) for sentence in sentence_vectors]
    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate traditional models
    results = {}

    # Logistic Regression
    model = LogisticRegression(max_iter=250)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    results["Logistic Regression"] = accuracy_score(y_valid, y_pred)

    # Linear SVM
    model = LinearSVC(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    results["Linear SVM"] = accuracy_score(y_valid, y_pred)

    # Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    results["Random Forest"] = accuracy_score(y_valid, y_pred)

    # Train and evaluate LSTM model
    # Compute average length of each training sample as estimation for a good maximum length
    average_length = 0
    for sentence in sentence_vectors:
        average_length += len(sentence)
    average_length /= len(sentence_vectors)
    max_len = int(average_length)
    sentence_vectors = pad_sequences(sentence_vectors, padding='post', dtype='float32', maxlen=max_len)

    # Create trainign and testing set
    X_train, X_valid = train_test_split(sentence_vectors, test_size = 0.2, random_state = 42)
    epochs, batch_size = lstm_params

    # Train either binary lstm model or multiclass lstm classifier depending
    if is_binary:
        y_train = np.array(y_train)
        y_valid = np.array(y_valid)

        model = binary_lstm_model(max_len)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        y_pred = model.predict(X_valid)
        y_pred = np.round(y_pred).flatten()
        print(f"LSTM Accuracy: {accuracy_score(y_valid, y_pred):.4f}")

    else:
        num_classes = len(set(y))
        y_train = to_categorical(y_train, num_classes=num_classes)  # convert to one hot encoding

        model = multiclass_lstm_model(max_len, num_classes)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        y_pred = model.predict(X_valid)
        y_pred = np.argmax(y_pred, axis=1)
        print(f"LSTM Accuracy: {accuracy_score(y_valid, y_pred):.4f}")

    # Print accuracy of traditional models
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")



# Call the process_and_train function for different datasets
dataset_files = [("financial_dataset.csv", False, 0, (8, 64)), ("hotel_dataset.csv", False, 1, (8, 512)), ("movie_dataset.csv", True, 0, (8, 512))]
for file_path, is_binary, offset_output, lstm_params in dataset_files:
    print(f"\nProcessing dataset: {file_path}")
    process_and_train(file_path, is_binary, offset_output, lstm_params)
