import pygame
from pygame.locals import *
pygame.init()

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch

from gan_mnist import Generator
from automata import DiffAutomata, Encoder

import time 

def display_image(img, surface):
    # 1 x 16 x 100 x 100
    val = img[0, :4].transpose(0, 2).cpu().numpy()
    alpha = np.clip(val[:, :, 3:4], 0, 1)
    img_ = np.clip((val[:, :, :3] * 255).astype(int), 0,255) * alpha + np.array([[[255, 255, 255]]]) * (1 - alpha)
    surf = pygame.surfarray.make_surface(img_)

    surf = pygame.transform.scale(surf, (500, 500))

    surface.blit(surf, (00, 00))

def display_image_monochrome(img, surface):
    # 1 x 16 x 100 x 100
    val = img[0, 0].cpu().numpy()
    img_ = 255 - np.repeat(np.expand_dims(np.clip((val * 255).astype(int), 0,255), -1), 3, axis = -1).transpose(1, 0, 2)
    #img_ = np.clip((val * 255).astype(int), 0,255).transpose(1, 0)

    surf = pygame.surfarray.make_surface(img_)

    surf = pygame.transform.scale(surf, (1000, 1000))

    surface.blit(surf, (00, 00))

def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("GPU is available!")
    else:
        dev = "cpu"
    
    device = torch.device(dev)

    da = DiffAutomata()
    enc = Encoder(28 * 28, 256, 50)
    gen = Generator(64, 120, 50)

    #load = torch.load("./models_da/da_autoencoder_3_2000.pt") 
    load = torch.load("./models/autoencoder_persistent_8000.pt") 
    load_gan = torch.load('./models/gan_48.pt')
    da.load_state_dict(load['model_state_dict'])
    enc.load_state_dict(load['encoder_state_dict'])
    gen.load_state_dict(load_gan['g_state_dict'])

    enc.to(device)
    da.to(device)
    gen.to(device)

    ds = torch.load("./MNIST/processed/training.pt")
    dt = TensorDataset(ds[0], ds[1])

    image = torch.zeros([1, 51, 200, 200]).to(device)
    #image[:, 0, 50, 50] = 1.
    #image[:, 1:, 50, 50] = torch.randn(1, 50).to(device)

    screen = pygame.display.set_mode((1000, 1000))

    c = 0
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                done = True
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    image = torch.zeros([1, 51, 200, 200]).to(device)
                elif event.key == pygame.K_s:
                    image = torch.zeros([1, 51, 200, 200]).to(device)

                    noise_vectors = torch.randn(49, 64).to(device)
                    latent_vectors = gen.forward(noise_vectors).view(7, 7, 50).transpose(2, 0)

                    image[:, 0, 10::30, 10::30] = 1.
                    image[:, 1:, 10::30, 10::30] = latent_vectors

            if event.type == pygame.MOUSEBUTTONDOWN:
                print(pygame.mouse.get_pressed())
                pos = pygame.mouse.get_pos()

                if pygame.mouse.get_pressed()[0]:
                    x = (pos[0]) // 5
                    y = (pos[1]) // 5

                    min_x = np.clip(x - 5, 0, 200)
                    max_x = np.clip(x + 5, 0, 200) 

                    min_y = np.clip(y - 5, 0, 200)
                    max_y = np.clip(y + 5, 0, 200) 

                    image[:, :, min_y : max_y, min_x : max_x] = 0.

                else:
                    x = (pos[0]) // 5
                    y = (pos[1]) // 5

                    # pick a random image and pass it through the encoder
                    #digit = (dt[np.random.randint(0, 50000)][0].type(torch.FloatTensor) / 255).view(1, 1, 28, 28).to(device)
                    noise_vector = torch.randn(1, 64).to(device)
                    latent_vector = gen.forward(noise_vector)

                    #latent_vector, _ = enc.forward(digit)

                    image[:, 0, y, x] = 1.
                    image[:, 1:, y, x] = latent_vector

        c += 1

        with torch.no_grad():
            screen.fill((255, 255, 255))
            a = time.time()

            display_image_monochrome(image, screen)
            #print(time.time() - a)
            time.sleep(0.025)
            mask = torch.randint(low = 0, high = 2, size = (1, 1, 200, 200)).to(device)
            update_mask = torch.ones([1, 1, 1, 1]).to(device)

            image = da.forward(image, mask, update_mask)
            #print(time.time() - a)

            pygame.display.flip()

if __name__ == "__main__":
    main()