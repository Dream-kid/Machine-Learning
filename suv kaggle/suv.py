import  pandas as pd
import  numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
import  math
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
dataset = pd.read_csv("suv_data.csv")
X= dataset.iloc[:,[2,3]].values
Y= dataset.iloc[:,4].values
print(Y)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train  = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
temp=classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import  accuracy_score

print(accuracy_score(y_test,y_pred))
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, y_pred))


#roc start





# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = classifier.predict_proba(X_test)
#lr_probs=temp
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
from matplotlib import pyplot
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()








