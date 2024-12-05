import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import csv


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

    # Remove stopwords (optional)
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Join the words back together
    cleaned_text = ' '.join(words)
    
    return cleaned_text

X = []
y = []

# Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


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



# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()  # TF-IDF implementation
# vectorizer = CountVectorizer()    # Bag of Words (BoW) implementation
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Train a logistic regression model
model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))