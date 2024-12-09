# Import necessary libraries
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
import gensim
import gensim.downloader as api
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import csv


# Download NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Instantiate stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Load pre-trained model
# word2vec = api.load('word2vec-google-news-300')
word2vec = gensim.models.KeyedVectors.load("word2vec-google-news-300.kv")

# Define the cleaning function
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove usernames
    text = re.sub(r'@\w+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove emojis
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes
        "\U0001F800-\U0001F8FF"  # supplemental arrows
        "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE
    )
    text = re.sub(emoji_pattern, '', text)

    # Tokenize the text into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Apply stemming
    # words = [stemmer.stem(word) for word in words]

    # Apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back together
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def vectorize_words(text):
    # Tokenize the text into words

    vector_list = []
    words = text.split()
    for word in words:
        if word in word2vec.key_to_index:
            vector_list.append(word2vec[word])
    return vector_list

def binary_lstm_model(length):
    model_lstm = Sequential()

    model_lstm.add(LSTM(128, input_shape=(length, word2vec.vector_size), return_sequences=False))

    model_lstm.add(Dropout(0.2))

    model_lstm.add(Dense(1, activation='sigmoid'))

    model_lstm.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model_lstm

def multiclass_lstm_model(length, n_classes):
    model_lstm = Sequential()

    model_lstm.add(LSTM(128, input_shape=(length, word2vec.vector_size), return_sequences=True))

    model_lstm.add(LSTM(64, return_sequences=False))
    model_lstm.add(Dropout(0.2))

    model_lstm.add(Dense(n_classes, activation='softmax'))

    model_lstm.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model_lstm

def read_movie_csv():
    y = []
    sentence_vectors = []
    
    # Load and preprocess data
    with open("movie_dataset.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for line in reader:
            cleaned_line = clean_text(line[0])
            word_vectors = vectorize_words(cleaned_line)
            sentence_vectors.append(word_vectors)
            y.append(0 if line[1] == 'negative' else 1)
    return (sentence_vectors, y, 2)

def read_financial_csv():
    y = []
    sentence_vectors = []
    
    # Load and preprocess data
    with open("financial_dataset.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for line in reader:
            cleaned_line = clean_text(line[0])
            word_vectors = vectorize_words(cleaned_line)
            sentence_vectors.append(word_vectors)
            y.append(int(line[1]))
    return (sentence_vectors, y, 3)

def read_hotel_csv():
    y = []
    sentence_vectors = []
    
    # Load and preprocess data
    with open("hotel_dataset.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for line in reader:
            text_portion = line[:-1]
            text = ""
            for segment in text_portion:
                text += segment
            
            cleaned_line = clean_text(text)
            word_vectors = vectorize_words(cleaned_line)
            sentence_vectors.append(word_vectors)
            y.append(int(line[-1]) - 1)
    return (sentence_vectors, y, 5)    

def deep_learn_binary():
    sentence_vectors, y, _ = read_movie_csv()
    average_size = 0
    for sentence in sentence_vectors:
        average_size += len(sentence)
    average_size /= len(sentence_vectors)
    max_len = int(average_size)

    y = np.array(y)
    padded_sequences = pad_sequences(sentence_vectors, padding='post', dtype='float32', maxlen=max_len)

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

    model_lstm = binary_lstm_model(max_len)
    model_lstm.fit(X_train, y_train, epochs=10, batch_size=128)

    y_pred_lstm = model_lstm.predict(X_valid)
    y_pred_lstm = np.round(y_pred_lstm)
    print(accuracy_score(y_valid, y_pred_lstm))

def deep_learn_multiclass(csv_func):
    sentence_vectors, y, num_classes = csv_func
    average_size = 0
    for sentence in sentence_vectors:
        average_size += len(sentence)
    average_size /= len(sentence_vectors)
    max_len = int(average_size)

    y = np.array(y)
    y = to_categorical(y, num_classes=num_classes)
    padded_sequences = pad_sequences(sentence_vectors, padding='post', dtype='float32', maxlen=max_len)

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

    model_lstm = multiclass_lstm_model(max_len, num_classes)
    model_lstm.fit(X_train, y_train, epochs=10, batch_size=512, use_multiprocessing=True)

    y_pred_lstm = model_lstm.predict(X_valid)
    y_pred_lstm = np.round(y_pred_lstm)
    print(accuracy_score(y_valid, y_pred_lstm))

def deep_learn(dataset):
    if dataset == 'movies':
        deep_learn_binary()
    elif dataset == 'hotels':
        deep_learn_multiclass(read_hotel_csv)
    elif dataset == 'financial':
        deep_learn_multiclass(read_financial_csv)

deep_learn('movies')
    

'''
# Function to process and train models for each dataset
def process_and_train(file_path):
    X, y = [], []
    sentence_vectors = []
    
    # Load and preprocess data
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for line in reader:
            cleaned_line = clean_text(line[0])
            word_vectors = vectorize_words(cleaned_line)
            sentence_vectors.append(word_vectors)
            y.append(line[1])
    
    padded_sequences = pad_sequences(sentence_vectors, padding='post', dtype='float32', maxlen=50)

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the data using TF-IDF
    # vectorizer = TfidfVectorizer()
    # X_train = vectorizer.fit_transform(X_train)
    # X_valid = vectorizer.transform(X_valid)

    # Train and evaluate models
    results = {}

    # Logistic Regression
    model_lr = LogisticRegression(max_iter=250)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_valid)
    results["Logistic Regression"] = accuracy_score(y_valid, y_pred_lr)

    # Naive Bayes
    model_nb = MultinomialNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_valid)
    results["Naive Bayes"] = accuracy_score(y_valid, y_pred_nb)

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

    return results

# Define dataset files (one file per dataset)
dataset_files = ["financial_dataset.csv", "hotel_dataset.csv", "movie_dataset.csv"]

# Run the process for each dataset file and print results
for i, file_path in enumerate(dataset_files, start=1):
    print(f"Dataset {i}: {file_path}")
    results = process_and_train(file_path)
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")
    print("-" * 50)
'''