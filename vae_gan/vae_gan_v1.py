import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cvimport_numpy as ci

train, _ = ci.data('', 1)


# Hyperparameters

lr = 3e-5
lr_ = 3e-5
epochs = 500
beta = 5
gamma = 15

# Generator --------------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    
    def __init__(self, input_channels = 3, output_channels = 1024, representation_size = 64):
        
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.features = nn.Sequential(
            # nc x 64 x 64
            nn.Conv2d(self.input_channels, representation_size, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(),
            # hidden_size x 32 x 32
            nn.Conv2d(representation_size, representation_size*2, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 2),
            nn.ReLU(),
            # hidden_size*2 x 16 x 16
            nn.Conv2d(representation_size*2, representation_size*4, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 4),
            nn.ReLU())
            # hidden_size*4 x 8 x 8
            
        self.mean = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
        self.logvar = nn.Sequential(
            nn.Linear(representation_size*4*8*8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))

        return mean, logvar
    
    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output

class Decoder(nn.Module):
    
    def __init__(self, input_size = 1024, representation_size = (256, 8, 8)):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
            # 256 x 8 x 8
        self.deconv1 = nn.ConvTranspose2d(representation_size[0], 256, 5, stride=2, padding=2) 
        self.act1 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU())
            # 256 x 16 x 16
        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
        self.act2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU())
            # 128 x 32 x 32
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2)
        self.act3 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU())
            # 32 x 64 x 64
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
            # 1 x 64 x 64
        self.activation = nn.Tanh()
            
    
    def forward(self, code):
        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 16, 16))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 32, 32))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 32, 64, 64))
        output = self.act3(output)
        output = self.deconv4(output, output_size=(bs, 1, 64, 64))
        output = self.activation(output)
        return output


class VAE_GAN_Generator(nn.Module):
    
    def __init__(self, input_channels = 3, hidden_size = 1024, representation_size=(256, 8, 8)):
        super(VAE_GAN_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size, representation_size)
        
    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()
        
        reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()

        reparametrized_noise = mean + std * reparametrized_noise

        rec_images = self.decoder(reparametrized_noise)
        
        return mean, logvar, rec_images


### Discriminator --------------------------------------------------------------------------------------------------------


class Discriminator(nn.Module):
    def __init__(self, input_channels = 3, representation_size=(256, 8, 8)):  
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.lth_features = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LeakyReLU(0.2))
        
        self.sigmoid_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        output = self.sigmoid_output(lth_rep)
        return output
    
    def similarity(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        return lth_rep


def visual(i, inp, recon, fake):

    inp = inp.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()
    fake = fake.detach().cpu().numpy()

    fig, ax = plt.subplots(nrows = 4, ncols = 3)

    for j in range(4):

        ax[j, 0].imshow(np.moveaxis(inp[j], 0, 2), cmap = 'gray')
        ax[j, 0].axis('off')
        ax[j, 1].imshow(np.moveaxis(recon[j], 0, 2), cmap = 'gray')
        ax[j, 1].axis('off')
        ax[j, 2].imshow(np.moveaxis(fake[j], 0, 2), cmap = 'gray')
        ax[j, 2].axis('off')

        if j == 0:
            ax[j, 0].set_title('Input')
            ax[j, 1].set_title('Recon')
            ax[j, 2].set_title('Gen')
    
    plt.savefig('/data_l77/shaikh/thesis/testing/' + str(i) + '.png', dpi = 300)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

G = VAE_GAN_Generator()
G.to(device)
D = Discriminator()
D.to(device)

criterion = nn.BCELoss()

optim_enc = torch.optim.RMSprop(G.encoder.parameters(), lr)
optim_dec = torch.optim.RMSprop(G.decoder.parameters(), lr)
optim_disc = torch.optim.RMSprop(D.parameters(), lr_)

real = torch.FloatTensor(np.array([[0],[0],[0],[0]])).cuda()
recon = torch.FloatTensor(np.array([[1],[1],[1],[1]])).cuda()
fake = torch.LongTensor(np.array([[2],[2],[2],[2]])).cuda()

enc = []
dec = []
disc = []

for i in range(epochs):

    tot_enc, tot_dec, tot_disc = 0, 0, 0

    for j in range(int(len(train)/4)):

        inp = train[j*4 : (j+1)*4]
        inp = torch.Tensor(inp)
        inp = inp.to(device)

        mean, logvar, rec_enc = G(inp)

        noise = Variable(torch.randn(4, 1024)).cuda()
        rec_noise = G.decoder(noise)

        # Train discriminator:

        out = D(inp)
        l_real = criterion(out, real.squeeze())
        out = D(rec_enc)
        l_recon = criterion(out, recon.squeeze())
        out = D(rec_noise)
        l_fake = criterion(out, recon.squeeze())
        disc_loss = l_real + l_recon + l_fake
        
        tot_disc += disc_loss.item()

        optim_disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()

        # Train decoder:

        mean, logvar, rec_enc = G(inp)

        noise = Variable(torch.randn(4, 1024)).cuda()
        rec_noise = G.decoder(noise)

        out = D(inp)
        l_real = criterion(out, real.squeeze())
        out = D(rec_enc)
        l_recon = criterion(out, recon.squeeze())
        out = D(rec_noise)
        l_fake = criterion(out, recon.squeeze())
        disc_loss = l_real + l_recon + l_fake

        similarity_rec_enc = D.similarity(rec_enc)
        similarity_data = D.similarity(inp)

        rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
        err_dec = gamma * rec_loss - disc_loss

        tot_dec += err_dec.item()

        optim_dec.zero_grad()
        err_dec.backward(retain_graph = True)
        optim_dec.step()

        # Train encoder:

        prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
        
        err_enc = prior_loss + beta * rec_loss

        tot_enc += err_enc.item()
        
        optim_enc.zero_grad()
        err_enc.backward()
        optim_enc.step()

        if j % 50 == 0:

            print('Encoder loss : {}, Decoder loss : {}, discriminator loss : {}'.format(err_enc.item(), err_dec.item(), disc_loss.item()))

    enc.append(tot_enc/int(len(train)/4))
    dec.append(tot_dec/int(len(train)/4))
    disc.append(tot_disc/int(len(train)/4))

    if i % 50 == 0:

        visual(i, inp, rec_enc, rec_noise)
        
        
    print('Epoch: {}, Encoder loss : {}, Decoder loss : {}, discriminator loss : {}'.format(i+1, tot_enc/int(len(train)/4), tot_dec/int(len(train)/4), tot_disc/int(len(train)/4)))


plt.plot(range(epochs), enc)
plt.plot(range(epochs), dec)
plt.plot(range(epochs), disc)
plt.show()


    
