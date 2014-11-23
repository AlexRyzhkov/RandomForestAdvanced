
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

import pandas as pd
import numpy as np

train = pd.read_csv('training.csv')
train.drop(['EventId'], axis=1, inplace=True)
train.drop(['Weight'], axis=1, inplace=True)
labels = train['Label'].values
print(labels)

ans = []
for c in labels:
	if c == 's':
		ans.append(1)
	else:
		ans.append(0)

test = train
test.drop(['Label'], axis=1, inplace=True)
test.to_csv('test.csv', index = False)

train['Label'] = ans
train.to_csv("train.csv", index = False)


