import torch
from torch import nn
from torch import optim 

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

import numpy as np

from automata import DiffAutomata, Encoder
class Generator(nn.Module):
    def __init__(self, noise_size, hidden_size, output_size):
        super().__init__()

        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear1 = nn.Linear(self.noise_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, noise):
        x = F.leaky_relu(self.linear1(noise))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)

    def forward(self, inp):
        x = F.leaky_relu(self.linear1(inp), negative_slope = 0.2)
        x = F.leaky_relu(self.linear2(x), negative_slope = 0.2)
        x = torch.sigmoid(self.linear3(x))

        return x

def main():
    plt.ion()

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("GPU is available!")
    else:
        dev = "cpu"
    
    #dev = "cpu"
    
    device = torch.device(dev)
    print(device)

    bs = 128
    noise_size = 64
    latent_size = 50

    g = Generator(noise_size, 120, latent_size)
    d = Discriminator(latent_size, 60)

    da = DiffAutomata()
    enc = Encoder(28 * 28, 256, 50)

    load = torch.load("./models/autoencoder_persistent_8000.pt")
    enc.load_state_dict(load['encoder_state_dict'])
    da.load_state_dict(load['model_state_dict'])

    ds = torch.load("./MNIST/processed/training.pt")
    dt = TensorDataset(ds[0], ds[1])
    dl = DataLoader(dt, batch_size = bs, shuffle = True, drop_last = True)

    g.to(device)
    d.to(device)
    da.to(device)
    enc.to(device)

    gsteps = 1
    dsteps = 1

    discriminator_optimizer = optim.Adam(d.parameters(), lr = 0.0002, betas = (0.5, 0.99), weight_decay = 0.02)
    generator_optimizer = optim.Adam(g.parameters(), lr = 0.0002, betas = (0.5, 0.99))

    for epoch in range (200):
        print("saving model")
        torch.save({
            "d_state_dict" : d.state_dict(),
            "g_state_dict" : g.state_dict(),
            "d_optimizer_state_dict" : discriminator_optimizer.state_dict(),
            "g_optimizer_state_dict" : generator_optimizer.state_dict(),
        }, "models_da/da_gan_%s.pt" % epoch)


        c = -1
        for X, Y in dl:
            c += 1

            for i in range (2):
                # generate bs x latent_size latent vectors from gaussian
                noise_vectors = torch.randn(bs, noise_size).to(device)

                # generate fake latent vectors
                g_latent = g.forward(noise_vectors)

                with torch.no_grad():
                    # pick bs real images from gaussian
                    images = (X.type(torch.FloatTensor) / 255.).view(bs, 1, 28, 28).to(device)

                    # get real latent vectors by encoding the images
                    r_latent, _ = enc.forward(images)  # bs x latent_size

                # to train discriminator, minimize log(fake), maximize log(real)
                fake = d.forward(g_latent.detach())
                real = d.forward(r_latent.detach())

                loss_d = -(torch.mean(torch.log(1 - fake)) + torch.mean(torch.log(real)))
            
                loss_d.backward()
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()
                d.zero_grad()
                g.zero_grad()

            for i in range (gsteps):
                avg = 0
                # now sample another bs x latent_size latent vectors from gaussian
                noise_vectors = torch.randn(bs, noise_size).to(device)
                g_latent_ = g.forward(noise_vectors)

                fake = d.forward(g_latent_)

                # to train generator, maximize log(fake)
                if c < 200 and epoch == 0:
                    loss_g = -torch.mean(torch.log(fake))
                else:
                    loss_g = torch.mean(torch.log(1 - fake))

                loss_g.backward()
                generator_optimizer.step()
                generator_optimizer.zero_grad()
                d.zero_grad()
                g.zero_grad()

                avg += loss_g

            print(loss_d, avg / gsteps, gsteps)

            if avg / gsteps > -0.7 and c > 500:
                gsteps = 2
            elif avg / gsteps > -0.8:
                gsteps = 1

            if c % 10 == 0:
                with torch.no_grad():
                    images = torch.zeros([bs, 51, 28, 28]).to(device)

                    noise_vectors = torch.randn(bs, noise_size).to(device)
                    g_latent = g.forward(noise_vectors)     # -> bs x latent_size 

                    images[:, 0, 14, 14] = 1. 
                    images[:, 1:, 14, 14] = g_latent 

                    for step in range (32):
                        mask = torch.randint(low = 0, high = 2, size = (bs, 1, 28, 28))
                        update_mask = torch.ones(bs, 1, 1, 1)

                        images = da.forward(images, mask.to(device), update_mask.to(device))

                    for i in range (20):
                        plt.subplot(4, 5, i + 1)
                        plt.cla()
                        plt.imshow(images[i, 0, :, :].view(28, 28).cpu().numpy())
                    plt.pause(0.0001)
            

if __name__ == "__main__":
    main()