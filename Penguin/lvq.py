"""
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
from sklearn import preprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

# Loading and examining the dataset
penguins_df = pd.read_csv("C:/Users/Casper/Desktop/Makine_Ogrenmesi/penguins/penguins.csv")

# Dropping rows with missing or invalid values
penguins_df = penguins_df.dropna()
penguins_clean = penguins_df[(penguins_df["flipper_length_mm"] > 0) & (penguins_df["flipper_length_mm"] < 4000)]

# One-hot encoding categorical variables
df = pd.get_dummies(penguins_clean).drop("sex_.", axis=1)

# 'sex' sütununu dikkate almayan DataFrame
veriler = penguins_clean.drop(columns=['sex'])

# Özellikleri ölçeklendirme
scaler = StandardScaler()
scaled_features_only = scaler.fit_transform(veriler)

etiket=penguins_clean.iloc[:,-1]

# Etiketleri 0 ve 1 olarak temsil eden bir DataFrame oluşturma
etikett = pd.DataFrame(etiket.map({'MALE': 1, 'FEMALE': 0}))
"""

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
from sklearn import preprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

class LVQ:
    def __init__(self, n_prototypes=5, learning_rate=0.01, max_epochs=100):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def initialize_prototypes(self, X, y):
        unique_labels = np.unique(y)
        self.prototypes = {}
        for label in unique_labels:
            label_data = X[y == label]
            if len(label_data) < self.n_prototypes:
                self.n_prototypes = len(label_data)  # Adjust the number of prototypes if not enough samples available
            random_indices = np.random.choice(len(label_data), size=self.n_prototypes, replace=False)
            self.prototypes[label] = label_data[random_indices]
            
    def find_closest_prototype(self, x):
        min_dist = float('inf')
        closest_prototype = None
        for label, prototypes in self.prototypes.items():
            for prototype in prototypes:
                dist = np.linalg.norm(x - prototype)
                if dist < min_dist:
                    min_dist = dist
                    closest_prototype = prototype
                    closest_label = label
        return closest_prototype, closest_label

    def update_prototypes(self, x, y, epoch):
        learning_rate = self.learning_rate * (1 - epoch / self.max_epochs)
        closest_prototype, closest_label = self.find_closest_prototype(x)
        if closest_label == y:
            closest_prototype += learning_rate * (x - closest_prototype)
        else:
            closest_prototype -= learning_rate * (x - closest_prototype)

    def fit(self, X, y):
        self.initialize_prototypes(X, y)
        for epoch in range(self.max_epochs):
            for x, label in zip(X, y):
                self.update_prototypes(x, label, epoch)

    def predict(self, X):
        predictions = []
        for x in X:
            closest_prototype, closest_label = self.find_closest_prototype(x)
            predictions.append(closest_label)
        return np.array(predictions)

# Loading and examining the dataset
penguins_df = pd.read_csv("C:/Users/Casper/Desktop/Makine_Ogrenmesi/penguins/penguins.csv")

# Dropping rows with missing or invalid values
penguins_df = penguins_df.dropna()
penguins_clean = penguins_df[(penguins_df["flipper_length_mm"] > 0) & (penguins_df["flipper_length_mm"] < 4000)]

# One-hot encoding categorical variables
#df = pd.get_dummies(penguins_clean).drop("sex_.", axis=1)

# 'sex' sütununu dikkate almayan DataFrame
veriler = penguins_clean.drop(columns=['sex'])

# Özellikleri ölçeklendirme
scaler = StandardScaler()
scaled_features_only = scaler.fit_transform(veriler)

etiket=penguins_clean.iloc[:,-1]

# LVQ modelini oluşturma ve eğitme
lvq = LVQ()
lvq.fit(scaled_features_only, etiket)

# Tahmin yapma
predictions = lvq.predict(scaled_features_only)

from sklearn.metrics import accuracy_score

# Tahminlerin doğruluk değerini hesapla
accuracy = accuracy_score(etiket, predictions)
print("Doğruluk Değeri:", accuracy)

# Boyut azaltma
pca = PCA(n_components=2)
penguins_pca = pca.fit_transform(scaled_features_only)

# LVQ tarafından öğrenilen kümeleri görselleştirme
plt.figure(figsize=(10, 6))

for label in np.unique(predictions):
    plt.scatter(penguins_pca[predictions == label][:, 0], 
                penguins_pca[predictions == label][:, 1],
                label=f'Cluster {label}')

plt.title('LVQ Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
