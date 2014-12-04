import numpy as np

def mse(preds, correct):
	error = 0.0
	preds = np.array(preds)
	correct = np.array(correct)
	error = np.mean(map(lambda x: x ** 2, (preds - correct)))
	return error