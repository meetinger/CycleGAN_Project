from testfunc import test

test(dataset="datasets/horse2zebra", batch=1, imSize=128, cuda=True, genXtoY = "minecraftday2night/weights/netXtoY.pth", genYtoX = "minecraftday2night/weights/netYtoX.pth")