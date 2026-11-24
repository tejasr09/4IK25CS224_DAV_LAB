import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv(Path(__file__).with_name('House_Price_dataset.csv'))
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

# OUTPUT PREVIEW (House_Price_dataset.csv; sampled rows: 100)
# Hierarchical cluster sizes: 0: 66, 1: 14, 2: 20.
# Mean price / bedrooms / baths / area size:
# 0: 11,235,909.09 / 2.94 / 2.92 / 5.07
# 1: 60,335,714.29 / 5.64 / 6.14 / 1.24
# 2: 26,460,000.00 / 4.75 / 5.25 / 10.45
# Displays the dendrogram and an area-vs-price cluster scatterplot.

