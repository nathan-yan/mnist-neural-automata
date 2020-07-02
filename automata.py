import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import matplotlib.pyplot as plt
import time

import os
if not os.path.exists('models_da'):
    os.makedirs('models_da')

kernel_sobel_x = torch.zeros([51, 51, 3, 3])
kernel_sobel_y = torch.zeros([51, 51, 3, 3])

for k in range (51):
    sobel_x = torch.zeros([51, 3, 3])
    sobel_x[k] = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ])

    kernel_sobel_x[k] = sobel_x

    sobel_y = torch.zeros([51, 3, 3])
    sobel_y[k] = torch.tensor([
        [-1., -2., -1.],
        [-0., 0., 0.],
        [1., 2., 1.]
    ])

    kernel_sobel_y[k] = sobel_y 

class DiffAutomata(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_x = nn.Conv2d(51, 51, 3, padding = 1, bias = False)
        self.conv_y = nn.Conv2d(51, 51, 3, padding = 1, bias = False)

        self.conv_1 = nn.Conv2d(153, 300, 1, 1, bias = False)
        self.conv_2 = nn.Conv2d(300, 51, 1, 1, bias = False)

        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)

        self.conv_2.weight = nn.Parameter(torch.zeros([51, 300, 1, 1]))
        
        self.conv_x.weight = nn.Parameter(kernel_sobel_x, requires_grad = False)
        self.conv_y.weight = nn.Parameter(kernel_sobel_y, requires_grad = False)
        
    def forward(self, inp, mask, update_mask):
        x = self.conv_x(inp)    # -> bs x c x w x h
        y = self.conv_y(inp)

        # stack them all together
        #print(x.shape)
        perception = torch.cat([x, y, inp], axis = 1)

        # 1 x 1 convolutions
        c1 = F.relu(self.conv_1(perception))
        c2 = self.conv_2(c1)

        delta = c2 * mask * update_mask

        # set last 50 hidden cells to 0
        delta[:, 25:, :, :] *= 0.

        out = inp + delta 

        # set cells with no alive to zero
        dead_mask = self.pool(out[:, 0:1, :, :]) > 0.1

        #print(out.shape, dead_mask.shape, update_mask.shape)
        out = out * dead_mask

        return out

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
    
        self.conv_1 = nn.Conv2d(1, 32, 3, padding = 3, bias = False)    # after pooling -> 16 x 16
        self.conv_2 = nn.Conv2d(32, 64, 3, padding = 1, bias = False)   # after pooling -> 8 x 8
        self.conv_3 = nn.Conv2d(64, 96, 3, padding = 0, bias = False)   # after pooling -> 4 x 4

        self.linear_1 = nn.Linear(96 * 9, self.hidden_size)

        self.means = nn.Linear(self.hidden_size, self.latent_size)
        self.variances = nn.Linear(self.hidden_size, self.latent_size)

    def forward(self, inp):
        x = F.leaky_relu(self.conv_1(inp), negative_slope = 0.1)
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)    # 14 x 14
        x = F.leaky_relu(self.conv_2(x), negative_slope = 0.1)
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)    # 7 x 7
        x = F.leaky_relu(self.conv_3(x), negative_slope = 0.1)  # 6 x 6
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)    # 3 x 3

        #print(x.shape)

        x = torch.flatten(x, start_dim = 1)

        x = self.linear_1(x)

        means = self.means(x)
        variances = torch.exp(self.variances(x))

        return means, variances

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

    da = DiffAutomata()
    enc = Encoder(28 * 28, 256, 50)

    enc.to(device)
    da.to(device)

    bs = 20

    optimizer = optim.Adam(list(da.parameters()) + list(enc.parameters()), lr = 0.0001)

    print(sum(p.numel() for p in da.parameters()))
    print(sum(p.numel() for p in enc.parameters()))
    

    ds = torch.load("./MNIST/processed/training.pt")
    dt = TensorDataset(ds[0], ds[1])
    dl = DataLoader(dt, batch_size = bs, shuffle = True, drop_last = True)

    for epoch in range (100):
        iteration = 0
        for X, Y in dl:
            
            X = (X.type(torch.FloatTensor) / 255.).view(bs, 1, 28, 28).to(device)
            
            target = X

            # get latent vectors
            sample = torch.randn(bs, 50).to(device)
            means, variances = enc.forward(target.detach())

            latent = means     # bs x 100

            steps = torch.tensor(np.random.randint(30, 36, bs))

            images = torch.zeros([bs, 51, 28, 28]).to(device)
            images[:, 0, 14, 14] = 1.      # set seed
            images[:, 1:, 14, 14] = latent

            for step in range (torch.max(steps)):
                mask = torch.randint(low = 0, high = 2, size = (bs, 1, 28, 28))
                update_mask = ((steps - step) > 0).view(bs, 1, 1, 1)

                images = da.forward(images, mask.to(device), update_mask.to(device))
                #print(images.shape)

            # compute loss
            # bs x 16 x 30 x 30
            diff = (images[:, :1, :, :] - target).view(bs, -1)
            loss = torch.mean(torch.sqrt(torch.sum(diff ** 2, axis = -1)))
            #loss += .1 * torch.mean(torch.sum(means ** 2, axis = 1) + torch.sum(variances ** 2, axis = 1) - torch.sum(torch.log(variances), axis = 1))

            print(iteration, loss)

            if iteration % 2 == 0:
                plt.subplot(3, 1, 1)
                plt.cla()
                plt.imshow(images[0].detach()[0].cpu().numpy())                
                plt.pause(0.001)
 
                plt.subplot(3, 1, 2)
                plt.cla()
                plt.plot(means[0].detach().cpu().numpy())
                plt.pause(0.001)
                
                plt.subplot(3, 1, 3)
                plt.cla()
                plt.plot(variances[0].detach().cpu().numpy())
                plt.pause(0.001)

            if iteration % 1000 == 0:
                print("saving model")
                torch.save({
                    "model_state_dict" : da.state_dict(),
                    "encoder_state_dict" : enc.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                }, "models_da/da_autoencoder_%s_%s.pt" % (epoch, iteration))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration += 1
            
if __name__ == "__main__":
    main()
