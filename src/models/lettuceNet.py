import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_stdlib_context

def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


def plot_filters(filters):
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    n_filters, ix = 3, 1
    for i in range(n_filters):
        # get the filter
        f = filters[i, :, :, :]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f.permute(1, 2, 0)[:, :, j], cmap='gray')
            ix += 1

    plt.show()


def plot_activations(activations, square=8, name="plot", limit=3):
    # square = 8
    ix = 0
    activations = activations.squeeze(0)
    # limit = 2
    fig = plt.figure()
    for _ in range(2):
        for _ in range(2):
            # specify subplot and turn of axis
            # ax = plt.subplot(square, square, ix+1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(activations[ix, :, :])
            plt.axis('off')
            plt.savefig("RN_Activation.png", bbox_inches='tight')
            plt.show()
            ix += 1
            if ix == limit:
                break

    # # show the figure
    # plt.show()
    # fig.savefig(f'./{name}.png', dpi=fig.dpi)


'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class LettuceNet(nn.Module):
    def __init__(self):
        super(LettuceNet, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        # self.resnet18[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # The resnet average pool layer before fc
        # self.avgpool = nn.AvgPool2d((10, 1))
        self.resnet_linear = nn.Linear(512, 1000)
        # Fully connected layer to regress the o/p of resnet -> 1 HR per clip
        self.fc_regression = nn.Linear(1000, 5)
        # self.rnn = nn.GRU(input_size=10, hidden_size=10)
        # self.fc = nn.Linear(10, 10)

    def forward(self, images, targets):
        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        x = self.resnet18(images)
        x = x.view(x.size(0), -1)
        # output dim: BSx1
        x = self.resnet_linear(x)
        out = self.fc_regression(x)

        return out

    def name(self):
        return "LettuceNet"


if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=False)
    print(resnet18)
