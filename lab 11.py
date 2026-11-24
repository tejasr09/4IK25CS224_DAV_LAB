import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv('/content/House_Price_dataset.csv')
df = df[df['purpose'] == 'For Sale']
df = df[['price', 'bedrooms', 'baths', 'Area Size']].dropna()
df = df[(df['bedrooms'] > 0) & (df['baths'] > 0)]
df = df[df['price'] < df['price'].quantile(0.99)]
df = df[df['Area Size'] < df['Area Size'].quantile(0.99)]
df = df.sample(n=100, random_state=42).reset_index(drop=True)
X = df[['price', 'bedrooms', 'baths', 'Area Size']]
X_scaled = StandardScaler().fit_transform(X)

linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(linked)
plt.title('Dendrogram -- House Segments')
plt.xlabel('Houses')
plt.ylabel('Distance')
plt.axhline(y=10, color='red', linestyle='--')
plt.show()

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['HC_Cluster'] = hc.fit_predict(X_scaled)

print(df['HC_Cluster'].value_counts())
print(df.groupby('HC_Cluster').mean().round(2))

plt.scatter(df['Area Size'], df['price'], c=df['HC_Cluster'], cmap='plasma')
plt.xlabel('Area Size')
plt.ylabel('Price')
plt.title('House Segments (Hierarchical Clustering)')
plt.show()

