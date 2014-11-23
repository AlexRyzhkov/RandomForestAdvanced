import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

train = pd.read_csv('data.csv')
labels = np.array(train[['Var2']].values)


train.drop(['Var2'], axis=1, inplace=True)

xtrain = np.array(train)

vals = np.zeros(10)
for i in range(10):
	X_train, X_test, Y_train, Y_test = train_test_split(xtrain, labels, test_size=0.2)
	clf = RandomForestClassifier(n_estimators = 1000, min_samples_split = 21)
	clf.fit(X_train, Y_train)

	pred = clf.predict(X_test)
	vals[i] = np.mean(pred != Y_test) 

print(np.mean(vals))

