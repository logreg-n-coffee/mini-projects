# Project: Clustering Fish using K-Means Clutstering 
# Dataset: http://jse.amstat.org/jse_data_archive.htm
# Goal: to cluster the fish according to species

# Perform the necessary imports
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Read the csv
df = pd.read_csv('datasets/fish.csv', header=None)

# Create arrays for features
samples = df.iloc[:, 1:6]
species = df.iloc[:, 0]

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
labels_species = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(labels_species['labels'], labels_species['species'])

# Display ct
print(ct)
