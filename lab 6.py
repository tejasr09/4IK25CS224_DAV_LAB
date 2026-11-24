import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Titanic-Dataset.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
features = df[['Pclass','Age','SibSp','Parch','Fare']]
print('Original data:\n', features.head())

minmax = MinMaxScaler()
normalized = minmax.fit_transform(features)
print('\nNormalized data (first 5 rows):\n',normalized[:5].round(3))
print('Normalized range:',normalized.min(),'to',normalized.max())

scaler = StandardScaler()
standardized = scaler.fit_transform(features)
print('\nStandardized data (first 5 rows):\n',standardized[:5].round(3))
print('Standardized mean:',standardized.mean().round(4),'std:',standardized.std().round(4))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(standardized)
print('\nShape before PCA:',standardized.shape,'| Shape after PCA',principal_components.shape)
print('Expected variance ratio:',pca.explained_variance_ratio_.round(4))
print('Total variance captured by 2 components:', round(sum(pca.explained_variance_ratio_)*100,2),'%')

colors=df['Survived'].map({1:'green',0:'red'})
plt.scatter(principal_components[:,0],principal_components[:,1],c=colors,alpha=0.6)
plt.scatter([],[],c='green',label='Survived')
plt.scatter([],[],c='red',label='Did not survive')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA: 5 features reduced to 2')
plt.legend()
plt.show()
