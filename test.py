import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, indim=484, hdim=87):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(indim, hdim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hdim, indim),
            nn.ReLU(),
        )

        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def plot_single_curve(x, y, x_label, y_label, title, save_path='figs/'):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.savefig(save_path + title + '.png')


if __name__ == '__main__':
    # train a one layer autoencoder of 484 dim one-hot vector to a 87 dim vector
    
    lossCNN__ = np.load('losses/lossCNN__.npy')
    plot_single_curve(range(len(lossCNN__)), lossCNN__, 'step', 'Loss', 'Loss of CNN', 'figs/')
    
    