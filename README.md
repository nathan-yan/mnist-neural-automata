# Growing MNIST digits with differential cellular automata.
This is based heavily on Mordvintsev et al. [https://distill.pub/2020/growing-ca/]. To generate novel digits you can run `automata_test.py`, which will use the pretrained models inside the `./models` directory.

To train your own network, run `automata.py` first to train the differential cellular automata and autoencoder to generate good looking MNIST digits. Then using the generated model, run `automata_persistent_mnist.py` to train the automata to generate persistent digits. Finally using the new generated model, run `gan_mnist.py` to train a Generative Adversarial Network on the latent vectors of the autoencoder. Finally using that model and the model from `automata_persistent_mnist.py`, you can run `automata_test.py`. The code right now is kinda messy so I'm working on cleaning it up and making the training pipeline easier.

Using this autoencoder/GAN technique is definitely not the best way of approaching it, and I'm hoping to train the whole thing end-to-end as a GAN to generate more realistic looking digits.

## Dependencies
- numpy>=1.16.0
- matplotlib>=3.0.0
- torch>=1.4.0
