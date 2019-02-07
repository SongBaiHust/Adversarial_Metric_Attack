import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
	if hasattr(m.bias, 'data'):
            init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        if hasattr(m.bias, 'data'):
            init.constant(m.bias.data, 0.0)

class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        num_bottleneck = 2048

        add_block += [nn.BatchNorm1d(num_bottleneck)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        self.fc0 = nn.Linear(num_bottleneck, class_num, bias = True)
        init.normal(self.fc0.weight.data, std=0.001)
        if hasattr(self.fc0.bias, 'data'):
            init.constant(self.fc0.bias.data, 0.0)

    def forward(self, x, if_train = True):
        x = self.model(x)
        if if_train == False:
            return x
        x = self.fc0(x)
        return x


class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_dense,self).__init__()
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function 
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        num_bottleneck = 1024
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.bn = add_block
        self.model = model_ft


        self.fc0 = nn.Linear(num_bottleneck, class_num, bias = True)
        init.normal(self.fc0.weight.data, std=0.001)
        if hasattr(self.fc0.bias, 'data'):
            init.constant(self.fc0.bias.data, 0.0)

    def forward(self, x, is_train=True):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.model.bn(x)
        if is_train == False:
            return x
        x = self.fc0(x)
        return x

# debug model structure
#net = ft_net(751)
#net = ft_net_dense(751)
#print(net)
#input = Variable(torch.FloatTensor(8, 3, 224, 224))
#output = net(input)
#print('net output size:')
#print(output.shape)
