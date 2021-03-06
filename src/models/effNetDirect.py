import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
import matplotlib.pyplot as plt
import config
from efficientnet_pytorch import EfficientNet

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


class LettuceNetEffDirect(nn.Module):
    def __init__(self):
        super(LettuceNetEffDirect, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        # resnet = models.resnet18(pretrained=True)
        # modules = list(resnet.children())[:-1]
        #
        # # inception = models.inception_v3(pretrained=True)
        # # modules = list(inception.children())[:-1]
        # self.inception = nn.Sequential(*modules)

        # self.resnet18 = nn.Sequential(*modules)
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        # Unfreeze model weights
        for param in self.model.parameters():
            param.requires_grad = True
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(4, 512)
        self.linear1 = nn.Linear(512, 512)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(512, len(config.FEATURES))
        self.linear_concat = nn.Linear(1024, 512)
        # self.linear_concat = nn.Linear(2056, 512)

        # self.linear = nn.Linear(4, 16)
        # self.linear1 = nn.Linear(16, 16)
        # self.activation = nn.LeakyReLU()
        # self.linear2 = nn.Linear(16, len(config.FEATURES))
        # self.linear_concat = nn.Linear(20, 16)
        self.resnet_linear = nn.Linear(1000, 512)
        # self.resnet_linear = nn.Linear(512, 64)
        # self.resnet_linear = nn.Linear(64, 16)
        # # self.linear_concat = nn.Linear(2056, 512)

        self.fc_regression3 = nn.Linear(512, 64)
        self.fc_regression4 = nn.Linear(64, 16)
        self.fc_regression5 = nn.Linear(20, 8)
        self.fc_regression6 = nn.Linear(8, len(config.FEATURES))

    def forward(self, images, targets, features):
        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        # x = self.linear(features)
        # x = self.activation(x)
        # x = self.linear1(x)
        # mid = self.activation(x)
        # x = self.linear1(x)
        # x = self.activation(x)
        # out = self.linear2(x)
        #
        img_feat = self.model(images)
        # img_feat = self.inception(images)
        # Add avg Pooling layer

        # the mentioned dim is the output dim of the CNN. eg: 512, 2048
        img_feat = img_feat.view((-1, 1000))
        img_feat = self.resnet_linear(img_feat)
        img_feat = self.activation(img_feat)
        img_feat = self.fc_regression3(img_feat)
        img_feat = self.activation(img_feat)
        img_feat = self.fc_regression4(img_feat)

        out = torch.cat((features, img_feat), 1)
        # out = self.linear_concat(out)
        # out = self.activation(out)
        # out = self.fc_regression3(out)
        # out = self.activation(out)
        # out = self.fc_regression4(out)
        out = self.fc_regression5(out)
        out = self.activation(out)
        out = self.fc_regression6(out)
        out = self.activation(out)

        return out

    def name(self):
        return "LettuceNet"


if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=False)
    print(resnet18)
