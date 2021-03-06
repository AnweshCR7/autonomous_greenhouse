import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
import matplotlib.pyplot as plt
import config

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


def viz_featuremaps(outputs):
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"../outputs/trained/layer_{num_layer}.png")
        # plt.show()
        plt.close()


def viz_filters(model_weights):
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig(f"../outputs/pretrained/filter{i}.png")
    # plt.show()


'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class LettuceNetPlus(nn.Module):
    def __init__(self):
        super(LettuceNetPlus, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        #
        # # inception = models.inception_v3(pretrained=True)
        # # modules = list(inception.children())[:-1]
        # self.inception = nn.Sequential(*modules)

        self.resnet18 = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.resnet18[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # The resnet average pool layer before fc
        # self.avgpool = nn.AvgPool2d((10, 1))
        # self.resnet_linear = nn.Linear(512, 1000)
        # # Fully connected layer to regress the o/p of resnet -> 1 HR per clip
        # self.fc_regression = nn.Linear(1000, 10)
        # # 10 ft from fc regression and 4 form concatenating the extra features.
        # self.fc_regression2 = nn.Linear(14, 20)
        # self.fc_regression3 = nn.Linear(20, 10)
        # self.fc_regression4 = nn.Linear(10, len(config.FEATURES))

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
        # self.resnet_linear = nn.Linear(512, 16)
        # # self.linear_concat = nn.Linear(2056, 512)

        self.fc_regression3 = nn.Linear(512, 64)
        self.fc_regression4 = nn.Linear(64, 16)
        self.fc_regression5 = nn.Linear(16, len(config.FEATURES))

    def forward(self, images, targets, features):
        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        x = self.linear(features)
        x = self.activation(x)
        x = self.linear1(x)
        mid = self.activation(x)
        # x = self.linear1(x)
        # x = self.activation(x)
        # out = self.linear2(x)
        #
        img_feat = self.resnet18(images)
        # img_feat = self.inception(images)
        # Add avg Pooling layer
        img_feat = self.avgpool(img_feat)
        model_children = list(self.resnet18.children())
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective weights to the list
        model_weights = []  # we will save the conv layer weights in this list
        conv_layers = []
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")
        # viz_filters(model_weights)
        results = [conv_layers[0](images[10].unsqueeze(0))]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results
        viz_featuremaps(results)
        # the mentioned dim is the output dim of the CNN. eg: 512, 2048
        img_feat = img_feat.view((-1, 512))
        out = torch.cat((mid, img_feat), 1)
        out = self.linear_concat(out)
        out = self.activation(out)
        out = self.fc_regression3(out)
        out = self.activation(out)
        out = self.fc_regression4(out)
        out = self.activation(out)
        out = self.fc_regression5(out)

        # img_feat = self.resnet18(images)
        # # Add avg Pooling layer
        # img_feat = self.avgpool(img_feat)
        # img_feat = img_feat.view((-1, 512))
        # img_feat = self.resnet_linear(img_feat)
        # # the mentioned dim is the output dim of the CNN. eg: 512, 2048
        # # img_feat = img_feat.view((-1, 16))
        # out = torch.cat((features, img_feat), 1)
        # out = self.linear_concat(out)
        # out = self.activation(out)
        # out = self.linear1(out)
        # out = self.activation(out)
        # out = self.final_out(out)

        return out

    def name(self):
        return "LettuceNet"


if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=False)
    print(resnet18)
