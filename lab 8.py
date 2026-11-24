import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report)
import matplotlib.pyplot as plt

df = pd.read_csv("/content/Titanic-Dataset.csv")
df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
df['Age']=df['Age'].fillna(df['Age'].median())
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked']=df['Embarked'].map({'S':0,'C':1,'Q':2})

x=df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

clf= LogisticRegression(max_iter=1000)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

print('Accuracy:', round(accuracy_score(y_test,y_pred), 3))
print('Precision:', round(precision_score(y_test,y_pred), 3))
print('Recall:', round(recall_score(y_test,y_pred), 3))
print('F1 Score:', round(f1_score(y_test,y_pred), 3))
print('\n',classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix:\n',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Survived', 'Survived'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

