# Project: Clustering Grain Seeds using K-Means, Hierarchical Clustering, and t-SNE
# Dataset: https://archive.ics.uci.edu/ml/datasets/seeds
# Goal: 1) to cluster the grain seeds and finetune the hyperparameter K 2) Hierarchical Clustering 3) t-SNE visualization

# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # KMeans
from scipy.cluster.hierarchy import linkage, dendrogram  # Hierarchical Clustering
from sklearn.manifold import TSNE  # TSNE


# Read the csv
df = pd.read_csv('datasets/seeds.csv', header=None)
varieties = open('datasets/seeds_varieties.csv').read().split('\n')

# Create arrays for features
samples = df.iloc[:, 0:6]

# Instantiate a KMeans classifier: model
kmeans = KMeans(n_clusters=3)

# Fit the classifier to the training data and predict the labels
labels = kmeans.fit_predict(samples)

# Fine-tuning the model - find k
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    kmeans = KMeans(n_clusters = k)
    
    # Fit model to samples
    kmeans.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Evaluating the grain clustering using cross tab
# Create a DataFrame with labels and varieties as columns: df
labels_varieties = pd.DataFrame({'labels': labels, 'varieties': varieties})
print(labels_varieties)

# Create crosstab: ct
ct = pd.crosstab(labels_varieties['labels'], labels_varieties['varieties'])

# Display ct
print(ct)

# Hierarchical clustering of the grain data

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

# t-SNE visualization of grain dataset 
# Create a TSNE instance: model
tsne = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = tsne.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()
