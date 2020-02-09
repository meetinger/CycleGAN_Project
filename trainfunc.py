import itertools
import torch
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

import main
import models
from utils import *
import sys
import datetime
import time
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt
import random
import os



def train(epochs=100, batch=1, dataset="datasets/horse2zebra", lr=0.0002, decrease=-1, lambdaCyc_x=10.0,
          lambdaCyc_y=10.0,
          lambdaIdentity=5.0, imSize=128, inputChannels=3, outputChannels=3, cuda=False, backupDelay=-1,
          cpus=-1, mnlSeed=False, seed=6, useBuffer=False):
    print(epochs, batch, dataset, lr, decrease, lambdaCyc_x, lambdaCyc_y, lambdaIdentity, imSize, inputChannels, outputChannels, cuda, backupDelay, cpus, mnlSeed, seed, useBuffer)
    if decrease <= 0:
        global decay
        decay = epochs // 2
    else:
        decay = decrease

    if cpus <= 0:
        global threads
        threads = multiprocessing.cpu_count()
    else:
        threads = cpus
    if backupDelay <= 0:
        global chechpntDelay
        chechpntDelay = epochs // 10
    else:
        chechpntDelay = backupDelay

    if mnlSeed:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    XtoY = models.Generator(inputChannels, outputChannels)
    YtoX = models.Generator(inputChannels, outputChannels)
    D_x = models.Discriminator(inputChannels)
    D_y = models.Discriminator(inputChannels)

    if cuda:
        XtoY.cuda()
        YtoX.cuda()
        D_x.cuda()
        D_y.cuda()

    XtoY.apply(initWeights)
    YtoX.apply(initWeights)
    D_x.apply(initWeights)
    D_y.apply(initWeights)

    criterionGAN = torch.nn.MSELoss()
    criterionCycle = torch.nn.L1Loss()
    criterionIdentity = torch.nn.L1Loss()

    optimizer_Genrators = torch.optim.Adam(itertools.chain(XtoY.parameters(), YtoX.parameters()),
                                           lr=lr, betas=(0.5, 0.999))
    optimizer_D_x = torch.optim.Adam(D_x.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_y = torch.optim.Adam(D_y.parameters(), lr=lr, betas=(0.5, 0.999))

    lrScheduler_Genrators = torch.optim.lr_scheduler.LambdaLR(optimizer_Genrators,
                                                              lr_lambda=LR_sched(epochs, decay).step)
    lrScheduler_D_x = torch.optim.lr_scheduler.LambdaLR(optimizer_D_x,
                                                        lr_lambda=LR_sched(epochs, decay).step)
    lrScheduler_D_y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_y,
                                                        lr_lambda=LR_sched(epochs, decay).step)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    input_x = Tensor(batch, inputChannels, imSize, imSize)
    input_y = Tensor(batch, inputChannels, imSize, imSize)
    targetReal = Variable(Tensor(batch).fill_(1.0), requires_grad=False)
    targetFake = Variable(Tensor(batch).fill_(0.0), requires_grad=False)

    if useBuffer:
        bufferFake_x = ImageBuffer()
        bufferFake_y = ImageBuffer()

    #   ---------------------- LOAD DATA
    transformList = [transforms.Resize(int(imSize * 1.12), Image.ANTIALIAS),
                     transforms.RandomCrop(imSize),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataset = DataLoader(LoadDataset(dataset, transformList=transformList),
                         batch_size=batch,
                         shuffle=True,
                         num_workers=threads)

    logger = LossLogger(epochs, len(dataset))

    if not os.path.exists('minecraftday2night/weights'):
        os.makedirs('minecraftday2night/weights')
    if not os.path.exists('minecraftday2night/weights'):
        os.makedirs('minecraftday2night/weights')
    if not os.path.exists('minecraftday2night/plots'):
        os.makedirs('minecraftday2night/plots')

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(dataset):
            currentBatch_x = Variable(input_x.copy_(batch['x']))
            currentBatch_y = Variable(input_y.copy_(batch['y']))

            fake_y = XtoY(currentBatch_x)
            fake_x = YtoX(currentBatch_y)

            optimizer_Genrators.zero_grad()

            lossGAN_G = criterionGAN(D_y(fake_y), targetReal)
            lossGAN_F = criterionGAN(D_x(fake_x), targetReal)

            recovered_x = YtoX(fake_y)
            recovered_y = XtoY(fake_x)
            lossCyc_x = criterionCycle(recovered_x, currentBatch_x)
            lossCyc_y = criterionCycle(recovered_y, currentBatch_y)
            lossCyc = lossCyc_x * lambdaCyc_x + lossCyc_y * lambdaCyc_y

            lossId_x = criterionIdentity(YtoX(currentBatch_x), currentBatch_x)
            lossId_y = criterionIdentity(XtoY(currentBatch_y), currentBatch_y)
            lossId = (lossId_x + lossId_y) * lambdaIdentity

            loss_Generators = lossGAN_G + lossGAN_F + lossCyc + lossId
            loss_Generators.backward()
            optimizer_Genrators.step()

            optimizer_D_x.zero_grad()
            if useBuffer:
                lossGAN_D_x = (criterionGAN(D_x(currentBatch_x), targetReal) + criterionGAN(
                    D_x(bufferFake_x.pushPop(fake_x).detach()), targetFake)) * 0.5
            else:
                lossGAN_D_x = (criterionGAN(D_x(currentBatch_x), targetReal) + criterionGAN(D_x(fake_x.detach()),
                                                                                            targetFake)) * 0.5
            lossGAN_D_x.backward()
            optimizer_D_x.step()

            optimizer_D_y.zero_grad()
            if useBuffer:
                lossGAN_D_y = (criterionGAN(D_y(currentBatch_y), targetReal) + criterionGAN(
                    D_y(bufferFake_y.pushPop(fake_y).detach()), targetFake)) * 0.5
            else:
                lossGAN_D_y = (criterionGAN(D_y(currentBatch_y), targetReal) + criterionGAN(D_y(fake_y.detach()),
                                                                                            targetFake)) * 0.5
            lossGAN_D_y.backward()
            optimizer_D_y.step()

            losses = {'loss_Gen': loss_Generators,
                      'loss_Gen_identity': lossId,
                      'loss_Gen_GAN': (lossGAN_G + lossGAN_F),
                      'loss_Gen_cycle': (lossCyc),
                      'loss_Disc': (lossGAN_D_x + lossGAN_D_y)}
            logger.log(epoch, i + 1, losses)

            main.window.setStatuses(epoch)

        lrScheduler_Genrators.step()
        lrScheduler_D_x.step()
        lrScheduler_D_y.step()

        if epoch % chechpntDelay == 0:
            label = '_ep' + str(epoch)
            torch.save(XtoY.state_dict(), 'minecraftday2night/weights/netXtoY' + label + '.pth')
            torch.save(YtoX.state_dict(), 'minecraftday2night/weights/netYtoX' + label + '.pth')
            torch.save(D_x.state_dict(), 'minecraftday2night/weights/netD_x' + label + '.pth')
            torch.save(D_y.state_dict(), 'minecraftday2night/weights/netD_y' + label + '.pth')

        torch.save(XtoY.state_dict(), 'minecraftday2night/weights/netXtoY.pth')
        torch.save(YtoX.state_dict(), 'minecraftday2night/weights/netYtoX.pth')
        torch.save(D_x.state_dict(), 'minecraftday2night/weights/netD_x.pth')
        torch.save(D_y.state_dict(), 'minecraftday2night/weights/netD_y.pth')

        logger.save()

    logger.plot()


