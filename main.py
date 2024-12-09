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
import csv

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Instantiate stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

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
    words = [stemmer.stem(word) for word in words]

    # Apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back together
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Define dataset files (one file per dataset)
dataset_files = ["sent_train1.csv", "sent_train2.csv", "sent_train3.csv"]

# Function to process and train models for each dataset
def process_and_train(file_path):
    X, y = [], []
    
    # Load and preprocess data
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for line in reader:
            X.append(clean_text(line[0]))
            y.append(line[1])
    
    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_valid = vectorizer.transform(X_valid)

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

# Run the process for each dataset file and print results
for i, file_path in enumerate(dataset_files, start=1):
    print(f"Dataset {i}: {file_path}")
    results = process_and_train(file_path)
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")
    print("-" * 50)