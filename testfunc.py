import torch
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
import models
import utils
import os
import sys


def test(dataset="datasets/horse2zebra", batch=1, imSize=128, inputChannels=3, outputChannels=3, cuda=False,
         cpus=-1, genXtoY="minecraftday2night/weights/netXtoY.pth", genYtoX="minecraftday2night/weights/netYtoX.pth"):
    print(dataset, batch, imSize, inputChannels, outputChannels, cuda, cpus, genXtoY, genYtoX)
    if cpus <= 0:
        global threads
        threads = multiprocessing.cpu_count()
    else:
        threads = cpus
    XtoY = models.Generator(inputChannels, outputChannels)
    YtoX = models.Generator(inputChannels, outputChannels)

    if cuda:
        XtoY.cuda()
        YtoX.cuda()

    XtoY.load_state_dict(torch.load(genXtoY))
    YtoX.load_state_dict(torch.load(genYtoX))

    XtoY.eval()
    YtoX.eval()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    input_x = Tensor(batch, inputChannels, imSize, imSize)
    input_y = Tensor(batch, outputChannels, imSize, imSize)

    transformList = [transforms.Resize(int(imSize), Image.ANTIALIAS),
                     transforms.CenterCrop(imSize),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataset = DataLoader(utils.LoadDataset(dataset, transformList=transformList, mode='test'),
                         batch_size=batch, shuffle=False, num_workers=threads)

    if not os.path.exists('minecraftday2night/x'):
        os.makedirs('minecraftday2night/x')
    if not os.path.exists('minecraftday2night/y'):
        os.makedirs('minecraftday2night/y')

    for i, batch in enumerate(dataset):
        currentBatch_x = Variable(input_x.copy_(batch['x']))
        currentBatch_y = Variable(input_y.copy_(batch['y']))

        # Generate minecraftday2night
        fake_y = 0.5 * (XtoY(currentBatch_x).data + 1.0)
        fake_x = 0.5 * (YtoX(currentBatch_y).data + 1.0)

        save_image(fake_x, 'minecraftday2night/x/%04d.jpg' % (i + 1))
        save_image(fake_y, 'minecraftday2night/y/%04d.jpg' % (i + 1))

        sys.stdout.write('\rGenerated %04d of %04d' % (i + 1, len(dataset)))

    sys.stdout.write('\n')
