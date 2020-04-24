import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import matplotlib.pyplot as plt
import time

from automata import DiffAutomata, Encoder

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

    load = torch.load("./models/autoencoder_14_2000.pt")
    da.load_state_dict(load['model_state_dict'])
    enc.load_state_dict(load['encoder_state_dict'])

    enc.to(device)
    da.to(device)

    bs = 20

    # only optimize the automata
    optimizer = optim.Adam(da.parameters(), lr = 0.0001)

    print(sum(p.numel() for p in da.parameters()))
    print(sum(p.numel() for p in enc.parameters()))

    ds = torch.load("./MNIST/processed/training.pt")
    dt = TensorDataset(ds[0], ds[1])
    dl = DataLoader(dt, batch_size = bs, shuffle = True, drop_last = True)

    pool = torch.zeros([200, 51, 28, 28]).to(device)
    target = torch.zeros([200, 1, 28, 28]).to(device)
    iterator = iter(dl)

    with torch.no_grad():
        for i in range (10):
            X, Y = iterator.next()
            target[i * 20 : i * 20 + 20] = (X.type(torch.FloatTensor) / 255.).view(bs, 1, 28, 28).to(device)
            
        means, variances = enc.forward(target.detach())
        latent = means

        # seed the pool
        pool[:, 0, 14, 14] = 1.
        pool[:, 1:, 14, 14] = latent

    for iteration in range (10000):
        if iteration % 3 == 0:
            with torch.no_grad():
                X, Y = iterator.next()
                X = (X.type(torch.FloatTensor) / 255.).view(bs, 1, 28, 28).to(device)
                
                # pick 20 random samples to replace with new seeds
                indices = np.random.randint(0, 200, bs)
                target[indices] = X.detach()

                latent, _ = enc.forward(X.detach())
                pool[indices] = torch.zeros([20, 51, 28, 28]).to(device)
                pool[indices, 0, 14, 14] = 1.
                pool[indices, 1:, 14, 14] = latent

        # select 20 random samples from the pool to do training on
        indices = np.random.randint(0, 200, bs)
        images = pool[indices].detach()

        if iteration % 10 == 0:
            for i in range (4):
                for j in range (5):
                    plt.subplot(4, 5, 1 + i * 5 + j)
                    plt.cla()
                    plt.imshow(images[i * 5 + j].detach().cpu().numpy()[0])
                plt.pause(0.0001)

        steps = torch.tensor(np.random.randint(30, 34, bs))
        for step in range (torch.max(steps)):
            mask = torch.randint(low = 0, high = 2, size = (bs, 1, 28, 28))
            update_mask = ((steps - step) > 0).view(bs, 1, 1, 1)

            images = da.forward(images, mask.to(device), update_mask.to(device))
        
        pool[indices] = images
        #print(images.shape)

        # compute loss
        # bs x 16 x 30 x 30
        diff = (images[:, :1, :, :] - target[indices]).view(bs, -1)
        loss = torch.mean(torch.sqrt(torch.sum(diff ** 2, axis = -1)))
        #loss += .1 * torch.mean(torch.sum(means ** 2, axis = 1) + torch.sum(variances ** 2, axis = 1) - torch.sum(torch.log(variances), axis = 1))

        print(iteration, loss)

        if iteration % 1000 == 0:
            print("saving model")
            torch.save({
                "model_state_dict" : da.state_dict(),
                "encoder_state_dict" : enc.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
            }, "models_da/da_autoencoder_persistent_%s.pt" % iteration)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

            
if __name__ == "__main__":
    main()