import  pandas as pd
import  numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
import  math
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
titanic_data = pd.read_csv('train.csv')
#print(titanic_data.head(10))
#print(len(titanic_data.Survived))
sns.countplot(x="Survived",hue="Age",data= titanic_data)
#plt.show()
titanic_data['Fare'].plot.hist(bins=20, figsize=(10,5))
#plt.show()
#print(titanic_data.info())
#print(titanic_data.isnull().sum())
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')
#plt.show()
sns.boxenplot(x="Pclass",y="Age",data=titanic_data)
#plt.show()
titanic_data.drop("Cabin",axis=1,inplace= True)
#titanic_data.dropna(inplace=True)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)
print(titanic_data.head(5))
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')
plt.show()
print(titanic_data.isnull().sum())
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark= pd.get_dummies(titanic_data['Embarked'],drop_first=True)
print(embark.head(5))
pcl= pd.get_dummies(titanic_data['Pclass'],drop_first=True)
print(pcl.head(5))
titanic_data = pd.concat([titanic_data,sex,embark,pcl],axis=1)
print(titanic_data.head(5))
titanic_data.drop(['Sex','Embarked','Ticket','PassengerId','Name','Pclass'],axis=1,inplace=True)
print(titanic_data.head(5))
X= titanic_data.drop("Survived",axis=1)
y= titanic_data["Survived"]
X_train, X_test,y_train,y_test=train_test_split(X,y,train_size=0.3,random_state=42)

logmodel = LogisticRegression()
#logmodel = GradientBoostingClassifier(learning_rate=0.3,max_depth=1)
logmodel.fit(X_train,y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
print(accuracy_score(y_test,prediction))
test = pd.read_csv('test.csv')



sns.heatmap(test.isnull(),yticklabels=False,cmap='viridis')
plt.show()
print(test.isnull().sum())
test.drop("Cabin",axis=1,inplace= True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
#test.dropna(inplace=True)
print(test.isnull().sum())


sex=pd.get_dummies(test['Sex'],drop_first=True)
embark= pd.get_dummies(test['Embarked'],drop_first=True)
print(embark.head(5))
pcl= pd.get_dummies(test['Pclass'],drop_first=True)
print(pcl.head(5))
test = pd.concat([test,sex,embark,pcl],axis=1)
print(test.head(5))
Id = test.PassengerId
test.drop(['Sex','Embarked','Ticket','PassengerId','Name','Pclass'],axis=1,inplace=True)
print(test.head(5))
final_predictions = logmodel.predict(test)

output = pd.DataFrame({'PassengerId':Id, 'Survived':final_predictions})
output.to_csv('submission.csv', index=False)
print(output.head())
print(len(output))
