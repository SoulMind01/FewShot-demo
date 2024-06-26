import torch
import torch.nn as nn
from torchvision import models
import math
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)


class Mish(nn.Module):
    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class GELU(nn.Module):
    def forward(self, input):
        return (
            0.5
            * input
            * (
                1
                + torch.tanh(
                    math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))
                )
            )
        )


def get_activation_function(activation_function: str):
    # ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu', 'swish', 'mish', 'gelu', 'softplus', 'softsign', 'hard_sigmoid']
    act_func_dict = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "swish": Swish(),
        "mish": Mish(),
        "gelu": GELU(),
        "softplus": nn.Softplus(),
        "softsign": nn.Softsign(),
        "hard_sigmoid": nn.Hardsigmoid(),
    }
    assert activation_function in act_func_dict.keys()

    return act_func_dict[activation_function]


class VGG16(nn.Module):

    def __init__(
        self,
        vector_size: int,
        biases: int,
        dataset_name: str,
        activation_function: str,
    ):
        inner_vector_size = {}
        inner_vector_size["fashion"] = 2304
        inner_vector_size["mnist"] = 2304
        inner_vector_size["cifar10"] = 4096
        super(VGG16, self).__init__()
        trained = True

        self.act = get_activation_function(activation_function)
        self.block1 = models.vgg16(pretrained=trained).features[0]
        self.block1.bias.requires_grad = False
        self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.bn7 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.bn8 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.block2 = models.vgg16(pretrained=trained).features[2]
        self.block3 = models.vgg16(pretrained=trained).features[4:6]
        self.block4 = models.vgg16(pretrained=trained).features[7]
        self.block5 = models.vgg16(pretrained=trained).features[9:11]
        self.block6 = models.vgg16(pretrained=trained).features[12]
        self.block7 = models.vgg16(pretrained=trained).features[14]
        self.block8 = models.vgg16(pretrained=trained).features[16]
        self.classifier = nn.Linear(
            inner_vector_size[dataset_name], vector_size, bias=biases
        )

        if biases == 0:
            self.block1.bias.requires_grad = False
            self.block2.bias.requires_grad = False
            self.block3[1].bias.requires_grad = False
            self.block4.bias.requires_grad = False
            self.block5[1].bias.requires_grad = False
            self.block6.bias.requires_grad = False
            self.block7.bias.requires_grad = False

    def forward(self, x):
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, dim=0)
        x = self.block1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.block2(x)
        x = self.act(x)
        x = self.block3(x)
        x = self.act(x)
        x = self.block4(x)
        x = self.act(x)
        x = self.block5(x)
        x = self.act(x)
        x = self.block6(x)
        x = self.act(x)
        x = self.block7(x)
        x = self.act(x)
        x = self.block8(x)
        x = self.act(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = nn.Sigmoid()(x)

        return x  # output
