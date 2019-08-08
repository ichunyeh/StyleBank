from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Sequential
import torchvision.models as models
import args

device = args.device
vgg16 = models.vgg16(pretrained=True).features.to(device).eval()


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.target = None
        self.mode = 'learn'
    def forward(self, input):
        if self.mode == 'loss':
            self.loss = self.weight * F.mse_loss(input, self.target)
        elif self.mode == 'learn':
            self.target = input.detach()
        return input

def gram_matrix(input):
    a, b, c, d = input.size() # a=batchs_size(=1); b=number of feature maps; (c,d)=dimension of a feature map(N=c*d)
    features = input.view(a*b, c*d) # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t()) # compute the gram product

    # normalize the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.targets = []
        self.mode = 'learn'
    def forward(self, input):
        G = gram_matrix(input)
        if self.mode == 'loss':
            self.loss = self.weight * F.mse_loss(G, self.target)
        elif self.mode == 'learn':
            self.target = G.detach()
        return input

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses:
content_layers = ['conv_9']
content_weight = {
    'conv_9': 1
}
style_layers = ['conv_2', 'conv_4', 'conv_6', 'conv_9']
style_weight = {
    'conv_2': 1,
    'conv_4': 1,
    'conv_6': 1,
    'conv_9': 1,

}
class LossNetwork(nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        cnn = deepcopy(vgg16)
        normalization = Normalization().to(device)

        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0 # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers: # add content loss
                content_loss = ContentLoss()
                content_loss.weight = content_weight[name]
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers: # add style loss
                style_loss = StyleLoss()
                style_loss.weight = style_weight[name]
                model.add_module('style_loss_{}'.format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model)-1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i+1)]

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

    def learn_content(self, input):
        for cl in self.content_losses:
            cl.mode = 'learn'
        for sl in self.style_losses:
            sl.mode = 'nop'
        self.model(input) # feed image to vgg16

    def learn_style(self, input):
        for cl in self.content_losses:
            cl.mode = 'nop'
        for sl in self.style_losses:
            sl.mode = 'learn'
        self.model(input) # feed image to vgg16

    def forward(self, input, content, style):
        self.learn_content(content)
        self.learn_style(style)

        for cl in self.content_losses:
            cl.mode = 'loss'
        for sl in self.style_losses:
            sl.mode = 'loss'
        self.model(input) # feed image to vgg16

        content_loss = 0
        style_loss = 0
        for cl in self.content_losses:
            content_loss += cl.loss
        for sl in self.style_losses:
            style_loss += sl.loss

        return content_loss, style_loss

class StyleBankNet(nn.Module):
    def __init__(self, total_style):
        super(StyleBankNet, self).__init__()
        self.total_style = total_style

        self.alpha = args.alpha

        self.encoder_net = Sequential(
            ConvLayer(3, int(32 * self.alpha), kernel_size=9, stride=1),
            nn.InstanceNorm2d(int(32 * self.alpha)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32 * self.alpha), int(64 * self.alpha), kernel_size=3, stride=2),
            nn.InstanceNorm2d(int(64 * self.alpha)),
            nn.ReLU(inplace=True),
            ConvLayer(int(64 * self.alpha), int(128 * self.alpha), kernel_size=3, stride=2),
            nn.InstanceNorm2d(int(128 * self.alpha)),
            nn.ReLU(inplace=True),
        )
        self.decoder_net = Sequential(
            UpsampleConvLayer(int(128 * self.alpha), int(64 * self.alpha), kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(int(64 * self.alpha)),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(int(64 * self.alpha), int(32 * self.alpha), kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(int(32 * self.alpha)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32 * self.alpha), 3, kernel_size=9, stride=1),
        )
        self.style_bank = nn.ModuleList([
            Sequential(
                ConvLayer(int(128 * self.alpha), int(128 * self.alpha), kernel_size=3, stride=1),
            )
            for i in range(total_style)
        ])

    def forward(self, X, style_id=None):
        z = self.encoder_net(X)
        if style_id is not None:
            new_z = []
            for idx, i in enumerate(style_id):
                zs = self.style_bank[i](z[idx].view(1, *z[idx].shape))
                new_z.append(zs)
            z = torch.cat(new_z, dim=0)
            # print(z.shape)
        return self.decoder_net(z)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        # if upsample:
        #     self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            # x_in = self.upsample_layer(x_in)
            x_in = nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
