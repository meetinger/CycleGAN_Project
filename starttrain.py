import trainfunc

trainfunc.train(epochs=100, batch=1, dataset="datasets/horse2zebra", imSize=64, cuda=True)