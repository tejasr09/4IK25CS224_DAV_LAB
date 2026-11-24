import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv('/content/House_Price_dataset.csv')
if 'purpose' in df.columns:
    df=df[df['purpose'].str.strip().str.lower() == 'for sale'].copy()

df['Area']=df['Area Size']
if 'Area Type' in df.columns:
    df.loc[df['Area Type']=='Kanal', 'Area']=df['Area Size']*20
elif 'Area Typr' in df.columns:
    df.loc[df['Area Typr']=='Kanal', 'Area']=df['Area Size']*20

if 'Price' in df.columns:
    df['price'] = df['Price']
df=df[(df['Area']>0) & (df['price']>0)]
if len(df) == 0:
    raise ValueError("The dataset is empty after filtering. Please verify the 'purpose' and 'price' columns.")

df=df[(df['Area']<df['Area'].quantile(0.99)) & (df['price'] < df['price'].quantile(0.99))]
df=df.sample(n=min(2000, len(df)), random_state=42).copy()

X=df[['Area', 'price']]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
kmeans=KMeans(n_clusters=5, random_state=42)
df['cluster']=kmeans.fit_predict(X_scaled)
df.head()
wcss=[]
for k in range(1,11):
  km=KMeans(n_clusters=k, random_state=42)
  km.fit(X_scaled)
  wcss.append(km.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
kmeans=KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster']=kmeans.fit_predict(X_scaled)
print(df['cluster'].value_counts().sort_index())
print(df.groupby('cluster')[['Area', 'price']].mean())
plt.scatter(df['Area'], df['price'], c=df['cluster'], cmap='viridis')
centers=scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], marker='X', s=200, c='red', label='Centroids')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Clustered Data'); plt.legend();
plt.show()


