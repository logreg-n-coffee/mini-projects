# Project: Building a Prototype Recommender System using "Non-negative matrix factorization" (NMF) and Cosine Similarity
# Dataset: https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia
# Goal: to build a prototype recommender system of Wikipedia articles


# Perform the necessary imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# Load the articles, titles, and words
df = pd.read_csv('datasets/wikipedia_vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())   # without it, there would be 13,000 columns (13,000 words in the file)
titles = list(df.columns)  # a list of articles
words = open('datasets/wikipedia_vocabulary_utf8.txt').read().split('\n')  # a list of the words that label the columns of the word-frequency array

# NMF applied to Wikipedia articles
# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
norm_features_titles = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = norm_features_titles.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = norm_features_titles.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())