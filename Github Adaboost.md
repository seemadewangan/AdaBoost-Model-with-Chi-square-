

import numpy as np

import pandas as pd

from numpy import mean

from numpy import std

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification


# load data

%matplotlib inline

raw_data = pd.read_csv('D:\dataclass.csv')

raw_data = pd.read_csv('D:\dataclass.csv', index_col = 0)



#First map the string values of diagnosis to integer.

def mapping(raw_data,feature):
    
	featureMap=dict()
    
	count=0
    
	for i in sorted(raw_data[feature].unique(),reverse=True):
        
		featureMap[i]=count
        
		count=count+1
    
	raw_data[feature]=raw_data[feature].map(featureMap)
    
	return raw_data

raw_data=mapping(raw_data,"project")

raw_data=mapping(raw_data,"package")

raw_data=mapping(raw_data,"complextype")

raw_data=mapping(raw_data,"NMO_type")

raw_data=mapping(raw_data,"NIM_type")

raw_data=mapping(raw_data,"NOC_type")

raw_data=mapping(raw_data,"WOC_type")



#apply Min-max normalization technique

def NormalizeData(data):
     
	return (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))



raw_data.sample(5)


#print(raw_data.columns)

X = raw_data.iloc[:,0:30]   #independent columns

y = raw_data.iloc[:,-1] 



#apply SelectKBest class to extract top 10 best features


from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)


#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score')) #print 10 best features



X, y = make_classificatio(n_samples=100,n_features=20,n_informative=15,
n_redundant=5, random_state=1)


# prepare the cross-validation procedure

cv = KFold(n_splits=10, random_state=1, shuffle=True)


#Appy AdaBoost Model

model = AdaBoostClassifier()

model.fit(X_train, y_train)

print(model)

y_pred = model.predict(X_test)

print(y_pred)


print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))



from sklearn.metrics import cohen_kappa_score

print('cohen_kappa_score:',cohen_kappa_score(y_test, y_pred))



from sklearn.metrics import matthews_corrcoef

print('matthews_corrcoef:',matthews_corrcoef(y_test, y_pred))

