import numpy as np
import numpy.random as nprd
import pandas as pd
from metrics import mse
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import networkx as nx
from itertools import izip
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)


class RFA_Classifier:
	def __init__(self, options):
		self.trainfile = options['i']
		self.testfile = options['t']
		self.ntrees = int(options['ntrees'])
		self.modelfile = options['m']
		self.outputfile = options['o']
		self.rho = int(options['rho'])
		self.maxLeafSize = 10
		self.labs = {}
		logging.info('Classifier created')

	# ==========================================================================

	def plot_tree(self, idx):
		nx.write_dot(self.trees[idx],'test.dot')
		pos = nx.graphviz_layout(self.trees[idx],prog='dot')
		nx.draw(self.trees[idx],pos,with_labels=False)
		plt.show()

	# ==========================================================================

	def getEntropy(self, D):
	    L = len(D)
	    valueList = np.unique(D)
	    numVals = len(valueList)
	    countVals = np.zeros(numVals)
	    Ent = 0
	    for idx, val in enumerate(valueList):
	        countVals[idx] = np.sum(D==val)
	        Ent += countVals[idx]*1.0/L*np.log2(L*1.0/countVals[idx])
	    return Ent


	def getMaxInfoGain(self, D,X,feat):
		EntWithoutSplit=self.getEntropy(D)
		feature=X[:,feat]
		L=len(feature)
		valueList=np.unique(feature)
		splits=np.diff(valueList)/2.0+valueList[:-1]
		maxGain=0
		bestSplit=0
		bestPart1=[]
		bestPart2=[]
		for split in splits:
		    Part1idx=np.argwhere(feature<=split)
		    Part2idx=np.argwhere(feature>split)
		    E1=self.getEntropy(D[Part1idx[:,0]])
		    l1=len(Part1idx)
		    E2=self.getEntropy(D[Part2idx[:,0]])
		    l2=len(Part2idx)
		    Gain=EntWithoutSplit-(l1*1.0/L*E1+l2*1.0/L*E2)
		    if Gain >= maxGain:
		        maxGain=Gain
		        bestSplit=split
		        bestPart1=Part1idx
		        bestPart2=Part2idx
		return maxGain,bestSplit,bestPart1,bestPart2

    # ==========================================================================

	def sample_data_for_tree(self, X, y):
		idx = nprd.randint(len(y), size = len(y))
		X_new = X[idx, :]
		Y_new = y[idx]
		return X_new, Y_new

	# ==========================================================================

	def get_feat_and_border(self, N, rho, X, Y):
		while (True):
			feat_num = nprd.randint(N, size = rho)
			maxGain,bestSplit,bestPart1,bestPart2 = self.getMaxInfoGain(Y,X,feat=feat_num)
			if (len(bestPart1) != 0 and len(bestPart2) != 0):
				break
		return feat_num, bestSplit, bestPart1, bestPart2 

	# ==========================================================================

	def split_data(self, bestPart1, bestPart2, X, Y):
		bestPart1 = bestPart1[:, 0]
		bestPart2 = bestPart2[:, 0]
		X_ar = (X[bestPart1, :], X[bestPart2, :])
		Y_ar = (Y[bestPart1], Y[bestPart2])
		return X_ar, Y_ar

	# ==========================================================================

	def tree_construct(self, cnt, Tree, X, Y, options):
		maxLeafSize = options['maxLeafSize']
		Nclass = options['Nclass']
		counts = np.zeros(Nclass)
		for i in xrange(Nclass):
			counts[i] = sum(Y == i)

		if (len(Y) <= maxLeafSize or any(counts == len(Y))):
			data = []
			for i in xrange(Nclass):
				data.append(counts[i] / float(len(Y)))
			Tree.node[cnt]['cls'] = data
			return cnt

		N = X.shape[1]

		feat_num, bestSplit, bestPart1, bestPart2 = self.get_feat_and_border(N, options['rho'], X, Y)
		div = [feat_num, bestSplit]
		X_ar, Y_ar = self.split_data(bestPart1, bestPart2, X, Y)

		Tree.node[cnt]['div'] = div
		
		cnt_new = cnt
		for (X_i, Y_i) in zip(X_ar, Y_ar):
			Tree.add_edge(cnt, cnt_new + 1)
			cnt_new = cnt_new + 1
			cnt_new = self.tree_construct(cnt_new, Tree, X_i, Y_i, options)
		return cnt_new


	# ==========================================================================
	def generate_data_cv(self, X, Y, ind, N_folds):
		sz = int(len(Y) / N_folds)
		logging.info("SZ = {}".format(sz))
		if (ind == 0):
			Xtrain = X[sz:len(Y), :]
			Ytrain = Y[sz:len(Y)]
			Xtest = X[0:sz, :]
			Ytest = Y[0:sz]
		elif (ind + 1 == N_folds):
			Xtrain = X[0:sz * ind, :]
			Ytrain = Y[0:sz * ind]
			Xtest = X[sz*ind:len(Y), :]
			Ytest = Y[sz*ind:len(Y)]
		else:
			Xtrain_1 = X[0:sz * ind, :]
			Ytrain_1 = Y[0:sz * ind]
			Xtrain_2 = X[sz*(ind + 1):len(Y), :]
			Ytrain_2 = Y[sz*(ind + 1):len(Y)]
			Xtrain = np.vstack((Xtrain_1, Xtrain_2))
			Ytrain = np.hstack((Ytrain_1, Ytrain_2))
			Xtest = X[sz*ind:sz*(ind + 1), :]
			Ytest = Y[sz*ind:sz*(ind + 1)]

		logging.info("SIZES = {}, {}, {}, {}".format(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape))
		return Xtrain, Ytrain, Xtest, Ytest


	def cross_validation(self, X_tr, Y_tr, N_folds):
		X = np.array(X_tr)
		Y = np.array(Y_tr)
		results = []
		for i in xrange(N_folds):
			Xtrain, Ytrain, Xtest, Ytest = self.generate_data_cv(X, Y, i, N_folds)
			self.fit(Xtrain, Ytrain)
			preds = self.predict(Xtest)
			results.append(mse(preds, Ytest))
			logging.info("CUR_Result = {}".format(results[i]))
			clf = RandomForestClassifier(n_estimators = 20, criterion = "entropy", min_samples_split = 10)
			clf.fit(Xtrain, Ytrain)
			preds = clf.predict_proba(Xtest)[:, 1]
			logging.info("ERROR_RF = {}".format(mse(preds, Ytest)))


		logging.info("Results = {}".format(results))
		logging.info("MeanError = {}".format(np.mean(results)))

	# ==========================================================================

	def fit(self, X_tr, Y_tr):
		Xtrain = np.array(X_tr)
		Ytrain = np.array(Y_tr)
		Nclass = len(np.unique(Y_tr))
		trees_array = []
		options = {
			'nTrees': self.ntrees,
			'maxLeafSize': self.maxLeafSize,
			'Nclass': Nclass,
			'rho': self.rho
		}
		for i in range(self.ntrees):
			X, Y = self.sample_data_for_tree(Xtrain, Ytrain)
			logging.info("{},     {}".format(X.shape, Y.shape))
			Tree = nx.DiGraph()
			Tree.add_node(0)
			self.tree_construct(0, Tree, X, Y, options)
			trees_array.append(Tree)
			logging.info("Fit tree {} complete!".format(i))
			pos = nx.graphviz_layout(Tree, prog='dot')

		self.trees = trees_array
		return self

	# ==========================================================================

	def TreePredict(self, tree, X_test):
		preds = np.zeros(X_test.shape[0])
		argmax = lambda array: max(izip(array, xrange(len(array))))[1]

		for i in xrange(preds.shape[0]):
			pos = 0
			while (True):
				try:
					div = tree.node[pos]['div']
				except:
					break
				if X_test[i, div[0]] <= div[1]:
					pos = min(tree[pos].keys())
				else:
					pos = max(tree[pos].keys())

			preds[i] = tree.node[pos]['cls'][1]
		return preds		

	def predict(self, X_test):
		X = np.array(X_test)
		preds = np.zeros(X.shape[0])

		for (p, tree) in enumerate(self.trees):
			preds = preds + self.TreePredict(tree, X)
			logging.info("Predict tree {} complete!".format(p))

		preds = preds / self.ntrees
		return preds

