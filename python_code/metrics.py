import numpy as np

def mse(preds, correct):
	error = 0.0
	preds = np.array(preds)
	correct = np.array(preds)
	error = np.mean(map((preds - correct), lambda x: x ** 2))
	return error