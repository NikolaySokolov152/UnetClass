import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn

from torchvision.models import (VGG11_BN_Weights,
                                resnet50, ResNet50_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights,                                
                                densenet121, DenseNet121_Weights
                                )
from collections import OrderedDict



VGG_TYPES = {'vgg11' : torchvision.models.vgg11, 
             'vgg11_bn' : torchvision.models.vgg11_bn, 
             'vgg13' : torchvision.models.vgg13, 
             'vgg13_bn' : torchvision.models.vgg13_bn, 
             'vgg16' : torchvision.models.vgg16, 
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19_bn' : torchvision.models.vgg19_bn, 
             'vgg19' : torchvision.models.vgg19}


class Custom_VGG(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 vgg_type='vgg11_bn', 
                 num_classes=1):
        super(Custom_VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(weights=VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None) ################################ костыль
        self.features = vgg.features

        # input change
        self.features[0] = nn.Conv2d(ipt_size[2],64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # init fully connected part of vgg        
        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))
        test_out = self.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_classes)
                                       )
        self._init_classifier_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

'''
# инициализация модели с прдедобученными весами
model_vgg = Custom_VGG(ipt_size=(240, 240, 4), pretrained=True)
model_vgg.to(device)
'''

class Custom_ResNet(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 resnet='resnet50', 
                 num_classes=1):
        super(Custom_ResNet, self).__init__()

        # load convolutional part of resnet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = torch.nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-1])]))
        # изменение входа классификатора
        self.features.conv1=nn.Conv2d(ipt_size[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)       
        # изменение выхода классификатора
        # init fully connected part of resnet        
        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))
        test_out = self.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 2048),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(2048, num_classes)
                                       )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

'''
# инициализация готовыми моделями
model_resnet = Custom_ResNet(ipt_size=(240, 240, 4), pretrained=True)
'''

class Custom_EfficientNet(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 efficientnet='efficientnet_b0', 
                 num_classes=1):
        super(Custom_EfficientNet, self).__init__()

        # load convolutional part of efficientnet
        efficientnet = efficientnet_b0(weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        # изменение входа классификатора
        self.features[0][0]=nn.Conv2d(ipt_size[2], 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)   

        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))    
        test_out =  self.avgpool(self.features(test_ipt))
        
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 1024),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(1024, num_classes)
                                       )
        self._init_classifier_weights()

    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


'''
# инициализация готовыми моделями
model_efficientnet = Custom_EfficientNet(ipt_size=(240, 240, 4), pretrained=True)
model_efficientnet.to(device)
'''




class Custom_DenseNet(nn.Module):
    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 densenet='densenet121', 
                 num_classes=1):
        super(Custom_DenseNet, self).__init__()

        # load convolutional part of efficientnet
        densenet = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)

        self.features = densenet.features

        # изменение входа классификатора
        self.features.conv0=nn.Conv2d(ipt_size[2], 64,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))    
        test_out =  self.features(test_ipt)

        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 1024),
                                         nn.ReLU(True),
                                         nn.Dropout(),
                                         nn.Linear(1024, 1024),
                                         nn.ReLU(True),
                                         nn.Dropout(),
                                         nn.Linear(1024, num_classes)
                                       )
        self._init_classifier_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

'''
# инициализация готовыми моделями
model_densenet = Custom_DenseNet(ipt_size=(240, 240, 4), pretrained=True)
model_densenet.to(device)
'''