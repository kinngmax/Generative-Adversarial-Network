import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import cvimport_numpy as ci

# Train data

##train, lab_train, test, lab_test = ci.data('/data_l77/shaikh/thesis/datasets/cell_images', 1)
###train, test, lab_test, val, lab_val = train[:50000], test[:5000], lab_test[:5000], test[5000:], lab_test[5000:]
##test, lab_test, val, lab_val = test[:3279], lab_test[:3729], test[3729:], lab_test[3729:]

train, lab_train = ci.data('', 0)
#train, lab_train, val, lab_val = imgs[:300], labs[:300], imgs[300:], labs[300:]

#----------- Model ----------------


epochs = 10
lr1 = 0.001
lr2 = 0.001
lr3 = 0.01
beta = 0.9
gamma = 1

#---------------- Encoder ---------------

class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size = 7)
        self.bn1 = nn.InstanceNorm2d(16)
        self.act1 = nn.ReLU()

        self.l1 = nn.Sequential(self.conv1, self.bn1, self.act1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size = 7)
        self.bn2 = nn.InstanceNorm2d(32)
        self.act2 = nn.ReLU()

        self.l2 = nn.Sequential(self.conv2, self.bn2, self.act2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size = 7)
        self.bn3 = nn.InstanceNorm2d(64)
        self.act3 = nn.ReLU()

        self.l3 = nn.Sequential(self.conv3, self.bn3, self.act3)

        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1600, 1024)

        self.pool = nn.MaxPool2d(2, 2, return_indices = True)

    def reparametrize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        eps = Variable(torch.randn(1, 1024))
        eps = eps.to(device)

        return mu + std * eps

    def forward(self, inp):

        idx = []

        l1 = self.l1(inp)
        l1, idx1 = self.pool(l1)

        idx.append(idx1)

        l2 = self.l2(l1)
        l2, idx2 = self.pool(l2)

        idx.append(idx2)

        l3 = self.l3(l2)
        l3, idx3 = self.pool(l3)

        idx.append(idx3)

        mu = self.fc1(l3.view(1, -1))
        logvar = self.fc2(l3.view(1, -1))

        z = self.reparametrize(mu, logvar)

        return l3, idx, mu, logvar, z

#---------------- Decoder ---------------

class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()

        self.fc = nn.Linear(1024, 1600)

        self.convt1 = nn.ConvTranspose2d(64, 32, kernel_size = 7)
        self.conv1a = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn1a = nn.InstanceNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn1b = nn.InstanceNorm2d(32)
        self.conv1c = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn1c = nn.InstanceNorm2d(32)
        self.act1 = nn.ReLU()

        self.l1 = nn.Sequential(self.convt1, self.conv1a, self.bn1a, self.conv1b, self.bn1b, self.conv1c, self.bn1c, self.act1)

        self.convt2 = nn.ConvTranspose2d(32, 16, kernel_size = 7)
        self.conv2a = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2a = nn.InstanceNorm2d(16)
        self.conv2b = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2b = nn.InstanceNorm2d(16)
        self.conv2c = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2c = nn.InstanceNorm2d(16)
        self.act2 = nn.ReLU()

        self.l2 = nn.Sequential(self.convt2, self.conv2a, self.bn2a, self.conv2b, self.bn2b, self.conv2c, self.bn2c, self.act2)

        self.convt3 = nn.ConvTranspose2d(16, 3, kernel_size = 7)
        self.conv3a = nn.Conv2d(3, 3, 3, padding = 1)
        self.bn3a = nn.InstanceNorm2d(3)
        self.conv3b = nn.Conv2d(3, 3, 3, padding = 1)
        self.bn3b = nn.InstanceNorm2d(3)
        self.conv3c = nn.Conv2d(3, 1, 3, padding = 1)
        self.act3 = nn.ReLU()

        self.l3 = nn.Sequential(self.convt3, self.conv3a, self.bn3a, self.conv3b, self.bn3b, self.conv3c, self.act3)

        self.unpool = nn.MaxUnpool2d(2, 2)

        self.fc = nn.Linear(1024, 1600)

    def forward(self, x, idx):

        x = self.fc(x)
        x = x.view(1, 64, 5, 5)

        x = self.unpool(x, idx[2])
        l1 = self.l1(x)

        l2 = self.unpool(l1, idx[1])
        l2 = self.l2(l2)

        l3= self.unpool(l2, idx[0])
        l3 = self.l3(l3)

        return l3

#------------- Discriminator ------------

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 7)
        self.bn1 = nn.InstanceNorm2d(16)
        self.act1 = nn.ReLU()

        self.l1 = nn.Sequential(self.conv1, self.bn1, self.act1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size = 7)
        self.bn2 = nn.InstanceNorm2d(32)
        self.act2 = nn.ReLU()

        self.l2 = nn.Sequential(self.conv2, self.bn2, self.act2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size = 7)
        self.bn3 = nn.InstanceNorm2d(64)
        self.act3 = nn.ReLU()

        self.l3 = nn.Sequential(self.conv3, self.bn3, self.act3)

        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 3)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inp):

        l1 = self.l1(inp)
        l1 = self.pool(l1)

        l2 = self.l2(l1)
        l2 = self.pool(l2)

        l3 = self.l3(l2)
        l3 = self.pool(l3)

        l3 = l3.view(1, -1)

        out = self.fc1(l3)
        out = self.fc2(out)
        out = self.act3(out)

        return out


class VAE(nn.Module):

    def __init__(self):

        super(VAE, self).__init__()

        self.e = Encoder()
        self.d = Decoder()
        self.mse = nn.MSELoss()

    def forward(self, x):

        hid, idx, mu, logvar, z  = self.e(x)
        out = self.d(z, idx)

        return out, mu, logvar

    def loss(self, mu, logvar, inp, recon):

        se = self.mse(recon.view(1, -1), inp.view(1, -1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return se + 2 * kl
        

MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

enc = Encoder()
enc.to(device)
dec = Decoder()
dec.to(device)
dis = Discriminator()
dis.to(device)

vae = VAE()
vae.to(device)

opt_vae = torch.optim.SGD(vae.parameters(), lr1)

opt_enc = torch.optim.SGD(enc.parameters(), lr1)
opt_dec = torch.optim.SGD(dec.parameters(), lr2)
opt_dis = torch.optim.SGD(dis.parameters(), lr3)


#---------------------------- Training -----------------------------------

##
##for i in range(epochs):
##
##    tot_enc, tot_dec, tot_dis = 0, 0, 0
##
##    for j in range(len(train)):
##
##        inp = train[j]
##        inp = torch.Tensor(inp)
##        inp = inp.to(device)
##        inp = inp.view(1, 1, 82, 82)
##
##        #---- gen. latent space z
##
##        hid, idx, mu, logvar, z = enc(inp)
##
##        #---- gen. reconstruction
##
##        recon = dec(z, idx)
##
####        #---- gen. recon from random vector
####
####        randm = torch.Tensor(torch.randn(1, 1024))
####        randm = randm.to(device)
####
####        gen = dec(randm, idx)
####
####        #---- discriminator
####
####        dis_real = dis(inp)
####        dis_recon = dis(recon)
####        dis_fake = dis(gen)
####
####        #---- update discriminator
####
####        opt_dis.zero_grad()
####
####        dis_loss = CE(dis_real, torch.cuda.LongTensor([0])) + CE(dis_recon, torch.cuda.LongTensor([1])) + CE(dis_fake, torch.cuda.LongTensor([2]))
####
####        dis_loss.backward(retain_graph = True)
####
####        opt_dis.step()
##
##        #---- update decoder
##
##        opt_dec.zero_grad()
##
##        dec_loss = gamma * MSE(recon, torch.cuda.FloatTensor(inp))# - (dis_loss)
##
##        dec_loss.backward(retain_graph = True)
##
##        opt_dec.step()
##
##        #----- update encoder
##
##        opt_enc.zero_grad()
##
##        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
##
##        enc_loss = MSE(recon, torch.cuda.FloatTensor(inp)) + beta * KL
##
##        enc_loss.backward()
##
##        opt_enc.step()
##
##        if j % 100 == 0:
##
##            print('Encoder loss : {}, Decoder loss : {}, Discriminator loss : {}'.format(enc_loss.item(), dec_loss.item(), 0))
##
##        tot_enc += enc_loss.item()
##        tot_dec += dec_loss.item()
##        #tot_dis += dis_loss.item()
##
##    avg_enc = tot_enc/len(train)
##    avg_dec = tot_dec/len(train)
##    #avg_dis = tot_dis/len(train)
##
##    print('Epoch : {}, Avg - Encoder loss : {}, Decoder loss : {}, Discriminator loss : {}'.format(
##        (i+1), enc_loss, dec_loss, 0))

for i in range(epochs):

    tot_enc, tot_dec, tot_dis = 0, 0, 0

    for j in range(len(train)):

        inp = train[j]
        inp = torch.Tensor(inp)
        inp = inp.to(device)
        inp = inp.view(1, 1, 82, 82)

        recon, mu, logvar = vae.forward(inp)

        opt_vae.zero_grad()
        
        loss = vae.loss(mu, logvar, inp, recon)

        loss.backward()

        opt_vae.step()

        tot_enc += loss.item()

        if j % 100 == 0:

            print('Loss : {}'.format(loss.item()))

    print('Epoch : {}, Avg : {}'.format(i+1, tot_enc/len(train)))

        
