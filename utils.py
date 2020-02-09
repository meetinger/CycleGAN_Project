import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import glob
import os
from PIL import Image
import argparse
import itertools
import torch
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import sys
import datetime
import time
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt
import random
import os


class LR_sched():
    def __init__(self, numEpochs, decayEpoch):
        self.numEpochs = numEpochs
        self.decayEpoch = decayEpoch

    def step(self, currentEpoch):
        return 1.0 - max(0, currentEpoch - self.decayEpoch) / (self.numEpochs - self.decayEpoch)


class LoadDataset(Dataset):
    def __init__(self, dataRoot, transformList, mode='train'):
        self.transform = transforms.Compose(transformList)
        self.fileNames_x = sorted(glob.glob(os.path.join(dataRoot, '%s/x' % mode) + '/*.*'))
        self.fileNames_y = sorted(glob.glob(os.path.join(dataRoot, '%s/y' % mode) + '/*.*'))
        self.mode = mode

    def __getitem__(self, index):
        item_x = self.transform(Image.open(self.fileNames_x[index % len(self.fileNames_x)]).convert('RGB'))
        if self.mode == 'train':
            item_y = self.transform(
                Image.open(self.fileNames_y[random.randint(0, len(self.fileNames_y) - 1)]).convert('RGB'))
        else:
            item_y = self.transform(Image.open(self.fileNames_y[index % len(self.fileNames_y)]).convert('RGB'))
        return {'x': item_x, 'y': item_y}

    def __len__(self):
        return max(len(self.fileNames_x), len(self.fileNames_y))


class ImageBuffer():
    def __init__(self, size=50):
        self.size = size
        self.bufferSize = 0
        self.buffer = []

    def pushPop(self, data):
        if self.size == 0:
            return data
        returnData = []
        for element in data:
            element = torch.unsqueeze(element.data, 0)
            if self.bufferSize < self.size:
                self.bufferSize += 1
                self.buffer.append(element)
                returnData.append(element)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.size - 1)
                    tmp = self.buffer[random_id].clone()
                    returnData.append(tmp)
                    self.buffer[random_id] = element
                else:
                    returnData.append(element)
        return torch.cat(returnData, 0)


class LossLogger():
    def __init__(self, numEpochs, numBatches):
        self.numEpochs = numEpochs
        self.numBatches = numBatches
        self.losses = {}
        self.timeStart = time.time()
        self.timeBatchAvg = 0

    def log(self, currentEpoch, currentBatch, losses):
        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] | ' % (currentEpoch, self.numEpochs, currentBatch, self.numBatches))
        for lossName in losses:
            if lossName not in self.losses:
                self.losses[lossName] = []
                self.losses[lossName].append(losses[lossName].item())
            else:
                if len(self.losses[lossName]) < currentEpoch:
                    self.losses[lossName].append(losses[lossName].item())
                else:
                    self.losses[lossName][-1] += losses[lossName].item()
            sys.stdout.write('%s: %.4f | ' % (lossName, self.losses[lossName][-1] / currentBatch))
            if currentBatch % self.numBatches == 0:
                self.losses[lossName][-1] *= 1. / currentBatch

        batchesDone = (currentEpoch - 1) * self.numBatches + currentBatch
        self.timeBatchAvg = (time.time() - self.timeStart) / float(batchesDone)
        batchesLeft = self.numEpochs * self.numBatches - batchesDone
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batchesLeft * self.timeBatchAvg)))


        if currentBatch % self.numBatches == 0:
            sys.stdout.write('\n')

    def plot(self):
        for lossName in self.losses:
            plt.figure()
            plt.plot(range(len(self.losses[lossName])), self.losses[lossName])
            plt.title(lossName)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig('minecraftday2night/plots/' + lossName + '.png')

    def save(self):
        df = pd.DataFrame.from_dict(self.losses)
        df.to_csv("minecraftday2night/plots/losses.csv")


def initWeights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
