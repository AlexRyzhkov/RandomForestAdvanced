def mse(preds, correct):
	error = 0.0
	for (x,y) in zip(preds, correct):
		error += (x - y) ** 2
	error = error / len(preds)
	return error