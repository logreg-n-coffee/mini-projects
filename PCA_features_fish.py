# Project: Variance of the PCA features in the fish dataset
# Dataset: http://jse.amstat.org/jse_data_archive.htm
# Goal: 1) to find the intrinsic dimensions of the fish dataset 2) to use PCA for dimensionality reduction of the fish measurements, 
# retaining only the 2 most important components.
# Result: PCA features 0 and 1 have significant variance, the intrinsic dimension of this dataset appears to be 2.

# Perform the necessary imports
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Read the csv
df = pd.read_csv('datasets/fish.csv', header=None)

# Create arrays for features
samples = df.iloc[:, 1:6]

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Scale the data and save the scaled_samples
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

# Fit the pca model to 'samples'
pca.fit(scaled_samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)