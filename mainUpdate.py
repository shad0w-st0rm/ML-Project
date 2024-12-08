import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  
from sklearn.svm import LinearSVC              
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datasets import load_dataset
import csv

# Download NTLK Data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  

# Instantiate stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define the cleaning function
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove usernames (for social media text) - like @username
    text = re.sub(r'@\w+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)  # This removes punctuation like: !.,?;:" etc.

    # Remove numbers if they are not necessary
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

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
        "]+", flags=re.UNICODE)
    
    # Use the regex to substitute emojis with an empty string
    text = re.sub(emoji_pattern, '', text)

    # Tokenize the text into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Apply stemming (optional - you can comment this out if you prefer only lemmatization)
    words = [stemmer.stem(word) for word in words]

    # Apply lemmatization (optional - you can apply lemmatization before/after stemming)
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back together
    cleaned_text = ' '.join(words)
    
    return cleaned_text

X = []
y = []

# Text preprocessing
stop_words = set(stopwords.words('english'))

# Loading data from CSV files (original code)
file = open("sent_valid.csv", "r", encoding="utf-8")
reader = csv.reader(file)
for line in reader:
    X.append(clean_text(line[0]))
    y.append(line[1])

file = open("sent_train.csv", "r", encoding="utf-8")
reader = csv.reader(file)
for line in reader:
    X.append(clean_text(line[0]))
    y.append(line[1])

# Vectorization using TF-IDF (original code)
vectorizer = TfidfVectorizer()  # TF-IDF implementation
# vectorizer = CountVectorizer() # BOW implementation (if needed)
X = vectorizer.fit_transform(X)

# Train-test split (original code)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model (original code)
model_lr = LogisticRegression(max_iter=250)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Train a Naive Bayes model
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# Train a Linear SVM model
model_svm = LinearSVC(max_iter=1000)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Train a Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

