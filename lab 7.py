import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
df = pd.read_csv("/content/Titanic-Dataset.csv")
df = df[['Fare', 'Pclass', 'Age', 'SibSp', 'Parch']]
df['Age'] = df['Age'].fillna(df['Age'].median())
df = df[df['Fare'] > 0]
X_simple = df[['Pclass']]
y = df['Fare']
X_tr, X_te, y_tr, y_te = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression().fit(X_tr, y_tr)
pred_simple = model_simple.predict(X_te)
print("intercept:", round(model_simple.intercept_, 2))
print("slope:", round(model_simple.coef_[0], 2))
print("R2:", round(r2_score(y_te, pred_simple), 3))
print('RMSE:', round(np.sqrt(mean_squared_error(y_te, pred_simple)), 2))
X_multi = df[['Pclass', 'Age', 'SibSp', 'Parch']]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_multi, y, test_size=0.2, random_state=42)
model_multi = LinearRegression().fit(X_tr2, y_tr2)
pred_multi = model_multi.predict(X_te2)
print("intercept:", round(model_multi.intercept_, 2))
for name, coef in zip(X_multi.columns, model_multi.coef_):
    print("coefficient of", name, ":", round(coef, 2))
print("R2:", round(r2_score(y_te2, pred_multi), 3))
print('RMSE:', round(np.sqrt(mean_squared_error(y_te2, pred_multi)), 2))
plt.scatter(X_te,y_te,alpha=0.5,label='Actual')
plt.plot(X_te,pred_simple,color='red',label='Regression line')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
residuals=y_te2-pred_multi
plt.scatter(pred_multi,residuals,alpha=0.5)
plt.axhline(0,color='red',linestyle='--')
plt.xlabel('Predicted Fare')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
sns.histplot(residuals,kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()
sns.heatmap(X_multi.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
