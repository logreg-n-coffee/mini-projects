# Project: Building a Prototype Sentiment Analyzer with Logistic Regression
# Dataset: please see the comments below
# Goal: to train a sentiment analyzer by using the positive reviews and negative reviews

import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
# create the set of stopwords
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# load the reviews (using BeautifulSoup to load the XML files)
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html5lib")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html5lib")
negative_reviews = negative_reviews.findAll('review_text')


# nltk's tokenizer creates noises so we need preprocess the text
def my_tokenizer(s):
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form; jumps -> jump
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))


# create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)  # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()  # normalize it before setting label
    x[-1] = label
    return x

# N is the total number of examples
N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i, :] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i, :] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
orig_reviews, data = shuffle(orig_reviews, data)

X = data[:, :-1]
Y = data[:, -1]

# last 100 rows will be test
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Test accuracy:", model.score(Xtest, Ytest))

# let's look at the weights for each word
# try it with different threshold values!
threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)

# check misclassified examples
preds = model.predict(X)
P = model.predict_proba(X)[:, 1]  # p(y = 1 | x)

# since there are many, just print the "most" wrong samples
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None
for i in range(N):
    p = P[i]
    y = Y[i]
    if y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
    elif y == 0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p

print("Most wrong positive review (prob = %s, pred = %s):" % (minP_whenYis1, wrong_positive_prediction))
print(wrong_positive_review)
print("Most wrong negative review (prob = %s, pred = %s):" % (maxP_whenYis0, wrong_negative_prediction))
print(wrong_negative_review)
