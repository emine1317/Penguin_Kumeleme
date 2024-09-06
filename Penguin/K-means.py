import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn import metrics
from time import time
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Loading and examining the dataset
penguins_df = pd.read_csv("C:/Users/Casper/Desktop/Makine_Ogrenmesi/penguins/penguins.csv")

penguins_df.boxplot()
plt.show()

# Dropping rows with missing or invalid values
penguins_df = penguins_df.dropna()
penguins_clean = penguins_df[(penguins_df["flipper_length_mm"] > 0) & (penguins_df["flipper_length_mm"] < 4000)]

# One-hot encoding categorical variables
df = pd.get_dummies(penguins_clean.drop("sex", axis=1))

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Performing PCA to reduce dimensionality
pca = PCA(n_components=None)
penguins_pca = pca.fit_transform(X)

# Finding the optimal number of components to explain 90% variance
n_components = np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.9) + 1
print("Number of components to explain 90% variance:", n_components)

pca = PCA(n_components=n_components)
penguins_pca = pca.fit_transform(X)

# Finding the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_pca)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Clustering using K-means with the optimal number of clusters
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(penguins_pca)

# Visualizing the clusters
plt.scatter(penguins_pca[:, 0], penguins_pca[:, 1], c=kmeans.labels_, cmap="viridis")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title(f"K-means Clustering (K={n_clusters})")
plt.show()

# Benchmarking K-means clustering with different initialization methods
def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    kmeans.fit(data)
    print(f'{name:<9}\t{time() - t0:.3f}s\t{kmeans.inertia_}'
          f'\t{metrics.homogeneity_score(labels, kmeans.labels_):.3f}'
          f'\t{metrics.completeness_score(labels, kmeans.labels_):.3f}'
          f'\t{metrics.v_measure_score(labels, kmeans.labels_):.3f}'
          f'\t{metrics.adjusted_rand_score(labels, kmeans.labels_):.3f}'
          f'\t{metrics.adjusted_mutual_info_score(labels, kmeans.labels_):.3f}'
          f'\t{metrics.silhouette_score(data, kmeans.labels_, metric="euclidean", sample_size=300):.3f}')

digits = load_digits()
data = digits.data
labels = digits.target

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

n_digits = len(np.unique(labels))
bench_k_means(KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0),
              name="k-means++", data=data, labels=labels)

bench_k_means(KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0),
              name="random", data=data, labels=labels)

pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based", data=data, labels=labels)

print(82 * "_")

