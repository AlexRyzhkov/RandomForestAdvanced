import numpy as np
import numpy.random as nprd
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx
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
		logging.info('Classifier created')

	# ==========================================================================

	def plot_tree(self, idx):
		pos = nx.graphviz_layout(G,prog='dot')
		nx.draw(self.trees[idx],pos,with_labels=False,arrows=False)
		plt.show()

	# ==========================================================================

	def sample_data_for_tree(self, X, y):
		idx = nprd.randint(len(y), size = len(y))
		logging.info("{}".format(type(X)))
		X_new = X[idx, :]
		Y_new = y[idx]
		return X_new, Y_new

	# ==========================================================================

	def get_feat_and_border(self, N, rho, X, Y):
		feat_num = nprd.randint(N, size = rho)
		X_new = X[:, feat_num]
		clf = LinearRegression()
		clf.fit(X_new, y, n_jobs = -1)
		return feat_num, clf

	# ==========================================================================

	def split_data(self, div, X, Y):
		X_new = X[:, div[0]]
		clf = div[1]
		spl = (np.round(clf.decision_function(X_new)) == 1.)
		
		nspl = (1 - spl == 1)
		X_ar = [X_new[nspl, :], ]
		Y_ar = [Y[nspl], Y[spl]]

		return X_ar, Y_ar

	# ==========================================================================

	def tree_construct(self, cnt, Tree, X, Y, options):
		logging.info("{}, {}, {}".format(cnt, X.shape, len(Y)))
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
			logging.info("{} exit".format(cnt))
			return (Tree, cnt + 1)

		N = X.shape[1]
		div = self.get_feat_and_border(N, options['rho'], X, Y)
		Tree.node[cnt]['div'] = div
		X_ar, Y_ar = self.split_data(div, X, Y)
		cnt_new = cnt
		for (X_i, Y_i) in xzip(X_ar, Y_ar):
			Tree.add_edge(cnt, cnt_new + 1)
			Tree, cnt_new = self.tree_construct(cnt_new + 1, Tree, X_i, Y_i, optons)
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
			X, Y = self.sample_data_for_tree(Xtrain, Ytrain)
			Tree = nx.DiGraph()
			Tree.add_node(0)
			Tree, cnt = self.tree_construct(0, Tree, X, Y, optons)
			trees_array.append(Tree)

		self.trees = trees_array
		return self
	# ==========================================================================		