from RFA_Classifier import RFA_Classifier
from RFA_Regressor import RFA_Regressor
from metrics import mse
from collections import defaultdict
import getopt
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)


def get_argv_options(cmd):
	opts = defaultdict(lambda:'')
	options, remainder = getopt.getopt(cmd, 'n:vk:vt:v', ['type=', 'i=', 'm=', 't=', 'ntrees=', 'rho=', "o="])
	for (arg, val) in options:
		opts[arg[2:]] = val
	for key in opts.keys():
		print((key,opts[key]))
	return opts

class RandomForestAdvanced:
	def __init__(self, cmdl_arguments):
		options = get_argv_options(cmdl_arguments)
		self.trainfile = options['i']
		self.testfile = options['t']
		self.ntrees = int(options['ntrees'])
		self.modelfile = options['m']
		self.outputfile = options['o']
		self.rho = int(options['rho'])
		self.type = options['type']
		if self.type == 'cls':
			self.model = RFA_Classifier(options)
		elif self.type == 'reg':
			self.model = RFA_Regressor(options)
		else:
			raise NameError("Unknown type value in command line arguments!!!")

	def read_data(self):
		self.train_data = pd.read_csv(self.trainfile)
		#print(self.train_data)
		self.train_labels = np.array(self.train_data['Y'].values)
		print(self.train_labels)
		self.train_data.drop('Y', axis=1, inplace=True)
		self.test_data = pd.read_csv(self.testfile)
		return self

	def check_results(self, correct_answers):
		logging.info("ERROR = {}".format(mse(self.predictions, correct_answers)))

	def save_predictions(self):
		preds = pd.DataFrame(self.predictions)
		preds.to_csv(self.outputfile, index = False)

	def compare_with_RF(self):
		clf = RandomForestClassifier(n_estimators = 20, criterion = "entropy", min_samples_split = 10)
		clf.fit(self.train_data, self.train_labels)
		preds = clf.predict(self.test_data)
		logging.info("ERROR_RF = {}".format(mse(preds, self.train_labels)))


	def fit_and_test(self):
		self.read_data()
		#self.model.fit(self.train_data, self.train_labels)
		#self.model.plot_tree(0)

		self.model.cross_validation(self.train_data, self.train_labels, 10)

		#self.predictions = self.model.predict(self.test_data)
		#self.check_results(self.train_labels)
		#self.compare_with_RF()
		#self.save_predictions()
		

	



		


