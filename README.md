# ML-Project

The purpose of this project is to test and compare the accuracies of various ML models such as Logistic Regression, Linear SVM, Random Forest, and LSTM neural networks for text classification tasks. We used three distinct datasets to do this.

We had 3 total datasets, which included a financial dataset that classified financial tweets into 3 different categories (bullish market, bearish market, or neutral), a hotel dataset that classified hotel reviews into 1 to 5-star ratings, and a movie review classifier that classified movie reviews as positive or negative. The first two datasets had multiclass classifications, while the third had binary classifications, meaning our program had to successfully classify both types of data.

To do this, we first cleaned and preprocessed the text by removing unwanted elements such as URLs, punctuation, numbers, and stopwords. These words were then lemmatized to their root forms. The cleaned text was later converted to a vector mapping representation using the word2vec model. For the simple classifier models, namely the logistic regression, linear SVM, and random forest models, we averaged the word2vec vector for each word in the input sentence to obtain a single vector as input for each training example. We designed two LSTM models, a binary classifier and a multiclass classifier as well, in which we used padded vector sequences so all inputs were the same length. Additionally, we had to convert the outputs for the multiclass classifiers into one hot encoded outputs rather than numerically (such as 1, 2, 3). For all of the models, the 5 star rating system had to be decremented to a 0-4 value for better results and ‘positive’ and ‘negative’ was changed into 1 and 0. Each of these models was trained with an 80-20 training-testing split. The accuracies were then outputted based on the testing set and were compared/evaluated across all datasets and models.

NOTE: The Google word2vec pretrained model we used is very large, so we could not upload it in our zip file. Instead run the below python code to be able to create a local copy and then our code should work.

```python 
import gensim.downloader as api
word2vec = api.load('word2vec-google-news-300')
word2vec.save('word2vec-google-news-300.kv')
```