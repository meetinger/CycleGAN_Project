from testfunc import test

test(dataset="datasets/horse2zebra", batch=1, imSize=128, cuda=True, genXtoY = "output/weights/netXtoY.pth", genYtoX = "output/weights/netYtoX.pth")