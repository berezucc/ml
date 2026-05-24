"""
Assignment 4: Clustering
Nikita Berezyuk
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

"""
0. Dataset
- This is a dataset of Spotify tracks over a range of 125 different genres. Each track has some audio features associated with it. \
    The data is in CSV format which is tabular and can be loaded quickly. 

Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data
"""
### Load the dataset
file_path = r'C:\Users\nikit\Desktop\Repositories\cps-803\Clustering\dataset.csv'
data = pd.read_csv(file_path).set_index('Unnamed: 0')

"""
1. Data Pre-processing
"""
### Check for any duplicate tracks and artists, caused by being refernced in different albums
# print(data.groupby(['track_name', 'artists']).size().reset_index(name='count').sort_values('count', ascending=False)) 
data = data.drop_duplicates(subset=['track_name', 'artists'], keep='first')
# Drop all categorial features, keeping only numerical
non_numerical_features = ['track_id', 'artists', 'album_name', 'track_name', 'track_genre', 'explicit']
metadata = data[non_numerical_features]
data = data.drop(non_numerical_features, axis=1)
# drop all tracks made up entirly of spoken words
data = data[data['speechiness'] < 0.66]
# Handle missing values
data = data.fillna(data.mean())

### IQR-based outlier removal for all numerical features
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

"""
2. Data Exploration
"""
### Compute correlation matrix (https://www.displayr.com/what-is-a-correlation-matrix/)
correlation_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# energy and loudness features have high correlation. Since energy is not an objectively measured metric, I will drop it.
# data = data.drop(['energy'], axis=1)

# reduce dimensionality with low impact feature
data = data.drop(['time_signature'], axis=1)

### Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Dropping any constant features (https://medium.com/nerd-for-tech/removing-constant-variables-feature-selection-463e2d6a30d9)
var_thr = VarianceThreshold(threshold=0.25) 
var_thr.fit(scaled_data)
feature_mask = var_thr.get_support()

# none of the features were dropped, meaning all the features have sufficient variance
# Feature Mask (True = Retained, False = Dropped)
print(f"Variance Feature Mask: {feature_mask}")

"""
3. Clustering
"""
### Determine the optimal number of clusters using the Elbow method
inertia = []
range_n_clusters = range(1, 25)
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

### Applying K-means with the optimal number of clusters from elbow method
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
data['Cluster'] = clusters

"""
4. Visualize Clusters
"""
### Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
reduced_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
reduced_df['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 8))
for cluster in range(optimal_clusters):
    cluster_data = reduced_df[reduced_df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')

plt.title('Clusters Visualized (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

"""
5. Summarize Clusters
"""
# Print the summary of each cluster
cluster_summary = data.groupby('Cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

### Display feature patterns within a sample of the clusters
to_plot = data.sample(1000)
sns.pairplot(to_plot, vars=['energy', 'acousticness', 'valence'], hue='Cluster', palette='Set1', corner=True)
plt.title("Cluster Visualization")
plt.show()

# Sample Songs from Each Cluster
data_with_metadata = pd.concat([metadata, data], axis=1)
for i in range(data['Cluster'].nunique()):
    print(f"Cluster {i}")
    print("=" * 9)
    sampled_songs = data_with_metadata[data_with_metadata['Cluster'] == i].sample(5, random_state=42)
    for index, row in sampled_songs.iterrows():
        print(f"{row['track_name']} - {row['artists']} ({row['track_genre']})")
    print()

"""
6. Evaluate Clustering Metrics
"""
# Compute the silhouette score for the clustering
silhouette_avg = silhouette_score(scaled_data, clusters)
ch_score = calinski_harabasz_score(scaled_data, clusters)
db_score = davies_bouldin_score(scaled_data, clusters)

print(f"Average Silhouette Score for {optimal_clusters} clusters: {silhouette_avg:.2f}")
print(f"Calinski-Harabasz Score: {ch_score:.2f}")
print(f"Davies-Bouldin Score: {db_score:.2f}")