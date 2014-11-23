import numpy as np
import numpy.random as nprd
import pandas as pd
import logging
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
		self.maxLeafSize = 20
		print('Classifier created')

	# ==========================================================================

	def sample_data_for_tree(X, y):
		idx = nprd.randint(len(y), size = len(y))
		X_new = X[idx, :]
		Y_new = y[idx]
		return X_new, Y_new

	# ==========================================================================

	def sample_data_for_tree(X, y):
		idx = nprd.randint(len(y), size = len(y))
		X_new = X[idx, :]
		Y_new = y[idx]
		return X_new, Y_new


	# ==========================================================================

	def get_feat_and_border(N, rho, X, Y):
		feat_num = nprd.randint(N, size = rho)
		X_new = X[:, feat_num]
		## WRITE CODE HERE
		return feat_num, coefs


	# ==========================================================================

	def tree_construct(cnt, Tree, X, Y, options):
		maxLeafSize = options['maxLeafSize']
		Nclass = option['Nclass']
		counts = np.zeros((Nclass, 1))
		for i in xrange(Nclass):
			counts[i] = sum(Y == i)

		if (X.shape[0] <= maxLeafSize or any(counts == len(Y))):
			data = []
			for i in xrange(Nclass):
				data.append(counts[i] / len(Y))
			Tree.node[cnt]['cls'] = data
			return (Tree, cnt + 1)

		N = X.shape[1]
		div = [el for el in get_feat_and_border(N, options['rho'], X, Y)]
		Tree.node[cnt]['div'] = div
		X_ar, Y_ar = split_data(div, X, Y)
		cnt_new = cnt
		for (X_i, Y_i) in zip(X_ar, Y_ar):
			Tree.add_edge(cnt, cnt_new + 1)
			Tree, cnt_new = tree_construct(cnt_new + 1, Tree, X_i, Y_i, optons)
		return (Tree, cnt)


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
			X, Y = sample_data_for_tree(X_train, Y_train)
			Tree = nx.DiGraph()
			Tree.add_node(0)
			Tree, cnt = tree_construct(0, Tree, X, Y, optons)
			trees_array.append(Tree)

		self.trees = "aaa bbb"
		return self
	# ==========================================================================		