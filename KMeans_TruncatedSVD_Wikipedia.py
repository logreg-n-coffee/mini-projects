# Project: Clustering Wikipedia Articles and Titles based on a Given TD-IDF Array in CSR Matrix Format with KMeans
# Dataset: https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia
# Goal: to cluster different Wikipedia articles

# Perform the necessary imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Load the articles and titles
df = pd.read_csv('datasets/wikipedia_vectors.csv', index_col=0)
articles = csr_matrix(df.transpose())   # without it, there would be 13,000 columns (13,000 words in the file)
titles = list(df.columns)

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
