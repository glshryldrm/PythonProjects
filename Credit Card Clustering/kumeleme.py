import pandas as pd

df = pd.read_csv("CC GENERAL.csv") #https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

df.head()

# Eksik verileri kontrol et (kontrol amaçlı)
print("Eksik değerler:\n", df.isnull().sum())

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# CUST_ID sütununu çıkarıyoruz
df_clean = df.drop(columns=['CUST_ID'])

# Eksik değerleri ortalama ile dolduralım
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df_clean)

# Veriyi normalize edelim
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

df_scaled[:5]  # İlk 5 örneğe bakalım
#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # 2 boyutlu indirgeme
data_pca = pca.fit_transform(df_scaled)
#%%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



optimal_k = 2
# K-Means ile tekrar kümeleme
kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(data_pca)

kmeans_labels = kmeans.fit_predict(data_pca)

# Küme sonuçlarını inceleyelim
plt.figure(figsize=(12, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
#%%
from sklearn.cluster import DBSCAN
optimal_eps=2.5
optimal_min_samples =5
# DBSCAN'i tekrar çalıştır
dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples,metric='manhattan').fit(data_pca)

dbscan_labels = dbscan.fit_predict(data_pca)

# DBSCAN sonuçlarını görselleştirelim
plt.figure(figsize=(12, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels, cmap='plasma')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
#%%
from sklearn.metrics import davies_bouldin_score

# Davies-Bouldin Index
dbi_kmeans = davies_bouldin_score(df_scaled, kmeans_labels)
dbi_dbscan = davies_bouldin_score(df_scaled, dbscan_labels)
print(f'Davies-Bouldin Index (K-Means): {dbi_kmeans}')
print(f'Davies-Bouldin Index (DBSCAN): {dbi_dbscan}')
#%%
from sklearn.metrics import silhouette_score

# Silhouette Score (K-Means ve DBSCAN için)
silhouette_kmeans = silhouette_score(df_scaled, kmeans_labels)
silhouette_dbscan = silhouette_score(df_scaled, dbscan_labels)

print(f'Silhouette Score (K-Means): {silhouette_kmeans}')
print(f'Silhouette Score (DBSCAN): {silhouette_dbscan}')
#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K değerini test et
silhouette_scores = []
for k in range(2, 10):  # 2'den 10'a kadar olan kümeleri deniyoruz
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Silhouette skorlarını görselleştir
plt.plot(range(2, 10), silhouette_scores)
plt.title('Optimal K Değerini Bulmak İçin Silhouette Score')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Silhouette Score')
plt.show()
#%%
from sklearn.neighbors import NearestNeighbors
import numpy as np

# En yakın komşuları kullanarak epsilon için görsel analiz
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(df_scaled)
distances, indices = neighbors_fit.kneighbors(df_scaled)
distances = np.sort(distances[:, 4], axis=0)
plt.plot(distances)
plt.title('k-NN ile Epsilon Değerini Bulmak')
plt.ylabel('Mesafe')
plt.xlabel('Veri Noktası Sırası')
plt.show()

