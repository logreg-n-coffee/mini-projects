# Project: Discovering Topics (Interpretable Features) in Wikipedia Articles with "Non-negative matrix factorization" (NMF)
# Dataset: https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia
# Goal: to extract the interpretable features (topics) from Wikipedia 

# Perform the necessary imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

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

# Print the NMF features
print(nmf_features.round(2))

# NMF features of the Wikipedia articles (examine the feature)
# NMF components represent topics -
# looking at hep b and hep c, both articles are reconstructed using mainly the 4th NMF component (feature 4 is the highest)
# Create a pandas DataFrame: nmf_features_titles
nmf_features_titles = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Hepatitis B'
print(nmf_features_titles.loc['Hepatitis B'])

# Print the row for 'Hepatitis C'
print(nmf_features_titles.loc['Hepatitis C'])

# Identify the topics based on the results 
# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 4: component
component = components_df.iloc[4, :]

# Print result of nlargest
print(component.nlargest())

