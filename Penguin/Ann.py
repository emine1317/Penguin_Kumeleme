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

# 'sex' sütununu dikkate almayan DataFrame
veriler = penguins_clean.drop(columns=['sex'])

# Özellikleri ölçeklendirme
scaler = StandardScaler()
scaled_features_only = scaler.fit_transform(veriler)

etiket=penguins_clean.iloc[:,-1]

# Etiketleri 0 ve 1 olarak temsil eden bir DataFrame oluşturma
etikett = pd.DataFrame(etiket.map({'MALE': 1, 'FEMALE': 0}))

x=scaled_features_only
y=etikett


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri setin i eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Yapay sinir ağı modelini oluşturalım
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Modeli eğitelim
mlp.fit(X_train, y_train)

# Test seti üzerinde tahmin yapalım
predictions = mlp.predict(X_test)

# Doğruluk skorunu hesaplayalım
accuracy = accuracy_score(y_test, predictions)
print("Model accuracy:", accuracy)