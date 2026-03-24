import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn

from collections import OrderedDict


VGG_TYPES = {'vgg11' : torchvision.models.vgg11, 
             'vgg11_bn' : torchvision.models.vgg11_bn, 
             'vgg13' : torchvision.models.vgg13, 
             'vgg13_bn' : torchvision.models.vgg13_bn, 
             'vgg16' : torchvision.models.vgg16, 
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19' : torchvision.models.vgg19,
             'vgg19_bn' : torchvision.models.vgg19_bn, 
             }

VGG_WEIGHTS = {'vgg11' : torchvision.models.VGG11_Weights, 
               'vgg11_bn' : torchvision.models.VGG11_BN_Weights, 
               'vgg13' : torchvision.models.VGG13_Weights, 
               'vgg13_bn' : torchvision.models.VGG13_BN_Weights, 
               'vgg16' : torchvision.models.VGG16_Weights, 
               'vgg16_bn' : torchvision.models.VGG16_BN_Weights,
               'vgg19' : torchvision.models.VGG19_Weights,
               'vgg19_bn' : torchvision.models.VGG19_BN_Weights, 
               }


class Custom_VGG(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 vgg_type='vgg11_bn', 
                 num_classes=1,
                 use_end_sigmoid=False):
        super(Custom_VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg = VGG_TYPES[vgg_type](weights=VGG_WEIGHTS[vgg_type].IMAGENET1K_V1 if pretrained else None)
        self.features = vgg.features

        #print(vgg)

        # input change
        output_channels = self.features[0].out_channels
        self.features[0] = nn.Conv2d(ipt_size[2], output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # init fully connected part of vgg        
        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))
        test_out = self.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(4096, num_classes)
                                       )
        self._init_classifier_weights()

        self.sigmoid = nn.Sigmoid() if use_end_sigmoid else None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x if self.sigmoid is None else self.sigmoid(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def block_requires_grad_features(self):
        ## блокировка обучения сборщика фич
        for param in self.features.parameters():
            param.requires_grad = False

        # из-за модификации инпута необходимо его перетренировать
        for param in self.features[0].parameters():
            param.requires_grad = True

'''
# инициализация модели с прдедобученными весами
model_vgg = Custom_VGG(ipt_size=(240, 240, 4), pretrained=True)
model_vgg.to(device)
'''

RESNET_TYPES = {'resnet18' : torchvision.models.resnet18, 
                'resnet34' : torchvision.models.resnet34,
                'resnet50' : torchvision.models.resnet50,
                'resnet101': torchvision.models.resnet101,
                'resnet152': torchvision.models.resnet152
                }

RESNET_WEIGHTS = {'resnet18' : torchvision.models.ResNet18_Weights, 
                  'resnet34' : torchvision.models.ResNet34_Weights,
                  'resnet50' : torchvision.models.ResNet50_Weights,
                  'resnet101': torchvision.models.ResNet101_Weights,
                  'resnet152': torchvision.models.ResNet152_Weights}

class Custom_ResNet(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 resnet_type='resnet50', 
                 num_classes=1,
                 use_end_sigmoid=False):
        super(Custom_ResNet, self).__init__()

        assert resnet_type in RESNET_TYPES, "Unknown resnet_type '{}'".format(resnet_type)

        # load convolutional part of resnet
        if pretrained:
            try:
                res_net_weights =RESNET_WEIGHTS[resnet_type].IMAGENET1K_V2
            except Exception:
                res_net_weights =RESNET_WEIGHTS[resnet_type].IMAGENET1K_V1
        else:
            res_net_weights = None

        resnet_model = RESNET_TYPES[resnet_type](weights=res_net_weights)

        #print(resnet_model)
        
        self.features = torch.nn.Sequential(OrderedDict([*(list(resnet_model.named_children())[:-1])]))
        # изменение входа классификатора
        output_channels = self.features.conv1.out_channels
        self.features.conv1=nn.Conv2d(ipt_size[2], output_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)       
        # изменение выхода классификатора
        # init fully connected part of resnet        
        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))
        test_out = self.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Linear(self.n_features, num_classes)
        
        self._init_classifier_weights()

        self.sigmoid = nn.Sigmoid() if use_end_sigmoid else None

    def _init_classifier_weights(self):
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.normal_(0, 0.01)
            self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x if self.sigmoid is None else self.sigmoid(x)
        return x

    def block_requires_grad_features(self):
        # блокировка обучения сборщика фич
        for param in self.features.bn1.parameters():
            param.requires_grad = False
        for param in self.features.relu.parameters():
            param.requires_grad = False
        for param in self.features.layer1.parameters():
            param.requires_grad = False
        for param in self.features.layer2.parameters():
            param.requires_grad = False
        for param in self.features.layer3.parameters():
            param.requires_grad = False
        for param in self.features.layer4.parameters():
            param.requires_grad = False


'''
# инициализация готовыми моделями
model_resnet = Custom_ResNet(ipt_size=(240, 240, 4), pretrained=True)
'''

EFFICIENTNET_TYPES = {'efficientnet_b0'  : torchvision.models.efficientnet_b0, 
                      'efficientnet_b1'  : torchvision.models.efficientnet_b1,
                      'efficientnet_b2'  : torchvision.models.efficientnet_b2,
                      'efficientnet_b3'  : torchvision.models.efficientnet_b3,
                      'efficientnet_b4'  : torchvision.models.efficientnet_b4,
                      'efficientnet_b5'  : torchvision.models.efficientnet_b5,
                      'efficientnet_b6'  : torchvision.models.efficientnet_b6,
                      'efficientnet_b7'  : torchvision.models.efficientnet_b7,
                      'efficientnet_v2_s': torchvision.models.efficientnet_v2_s,
                      'efficientnet_v2_m': torchvision.models.efficientnet_v2_m,
                      'efficientnet_v2_l': torchvision.models.efficientnet_v2_l
                    }

EFFICIENTNET_WEIGHTS = {'efficientnet_b0'  : torchvision.models.EfficientNet_B0_Weights, 
                        'efficientnet_b1'  : torchvision.models.EfficientNet_B1_Weights,
                        'efficientnet_b2'  : torchvision.models.EfficientNet_B2_Weights,
                        'efficientnet_b3'  : torchvision.models.EfficientNet_B3_Weights,
                        'efficientnet_b4'  : torchvision.models.EfficientNet_B4_Weights,
                        'efficientnet_b5'  : torchvision.models.EfficientNet_B5_Weights,
                        'efficientnet_b6'  : torchvision.models.EfficientNet_B6_Weights,
                        'efficientnet_b7'  : torchvision.models.EfficientNet_B7_Weights,
                        'efficientnet_v2_s': torchvision.models.EfficientNet_V2_S_Weights,
                        'efficientnet_v2_m': torchvision.models.EfficientNet_V2_M_Weights,
                        'efficientnet_v2_l': torchvision.models.EfficientNet_V2_L_Weights
                    }


class Custom_EfficientNet(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 efficientnet_type='efficientnet_b0', 
                 num_classes=1,
                 use_end_sigmoid=False):
        super(Custom_EfficientNet, self).__init__()
        
        assert efficientnet_type in EFFICIENTNET_TYPES, "Unknown efficientnet_type '{}'".format(efficientnet_type)

        # load convolutional part of efficientnet
        efficientnet_model = EFFICIENTNET_TYPES[efficientnet_type](weights = EFFICIENTNET_WEIGHTS[efficientnet_type].IMAGENET1K_V1 if pretrained else None)

        #print(efficientnet_model)
        #print(efficientnet_model)
        self.features = efficientnet_model.features
        self.avgpool = efficientnet_model.avgpool

        # изменение входа классификатора
        output_channels = self.features[0][0].out_channels
        self.features[0][0]=nn.Conv2d(ipt_size[2], output_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)   

        #print(self.features)

        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))    
        test_out =  self.avgpool(self.features(test_ipt))
        
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(
                                        nn.Dropout(p=0.5, inplace=True),
                                        nn.Linear(self.n_features, num_classes)
                                       )
        self._init_classifier_weights()

        self.sigmoid = nn.Sigmoid() if use_end_sigmoid else None
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x if self.sigmoid is None else self.sigmoid(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def block_requires_grad_features(self):
        # блокировка обучения сборщика фич
        for param in self.features.parameters():
            param.requires_grad = False

        # из-за модификации инпута необходимо его перетренировать
        for param in self.features[0].parameters():
            param.requires_grad = True



'''
# инициализация готовыми моделями
model_efficientnet = Custom_EfficientNet(ipt_size=(240, 240, 4), pretrained=True)
model_efficientnet.to(device)
'''


DENSENET_TYPES = {'densenet121': torchvision.models.densenet121, 
                  'densenet161': torchvision.models.densenet161,
                  'densenet169': torchvision.models.densenet169,
                  'densenet201': torchvision.models.densenet201
                 }

DENSENET_WEIGHTS = {'densenet121': torchvision.models.DenseNet121_Weights, 
                    'densenet161': torchvision.models.DenseNet161_Weights,
                    'densenet169': torchvision.models.DenseNet169_Weights,
                    'densenet201': torchvision.models.DenseNet201_Weights
                   }

class Custom_DenseNet(nn.Module):
    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 densenet_type='densenet121', 
                 num_classes=1,
                 use_end_sigmoid=False):
        super(Custom_DenseNet, self).__init__()
        
        assert densenet_type in DENSENET_TYPES, "Unknown densenet_type '{}'".format(densenet_type)

        # load convolutional part of efficientnet
        densenet = DENSENET_TYPES[densenet_type](weights = DENSENET_WEIGHTS[densenet_type].IMAGENET1K_V1 if pretrained else None)

        #print(densenet)

        self.features = densenet.features

        # изменение входа классификатора
        output_channels = self.features.conv0.out_channels
        self.features.conv0=nn.Conv2d(ipt_size[2], output_channels,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))    
        test_out =  self.features(test_ipt)

        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Linear(self.n_features, num_classes)
        self._init_classifier_weights()

        self.sigmoid = nn.Sigmoid() if use_end_sigmoid else None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x if self.sigmoid is None else self.sigmoid(x)
        return x

    def _init_classifier_weights(self):
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.normal_(0, 0.01)
            self.classifier.bias.data.zero_()

    def block_requires_grad_features(self):
        # блокировка обучения сборщика фич
        for param in self.features.parameters():
            param.requires_grad = False

        # из-за модификации инпута необходимо его перетренировать
        for param in self.features.conv0.parameters():
            param.requires_grad = True

'''
# инициализация готовыми моделями
model_densenet = Custom_DenseNet(ipt_size=(240, 240, 4), pretrained=True)
model_densenet.to(device)
'''


class Custom_Googlenet(nn.Module):

    def __init__(self,
                 ipt_size=(240, 240, 4),
                 pretrained=True,
                 googlenet_type="default",
                 num_classes=1,
                 use_end_sigmoid=False):
        super(Custom_Googlenet, self).__init__()

        # load convolutional part of efficientnet
        model_googlenet = torchvision.models.googlenet(weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1) #if pretrained else None) # torchvision.models.GoogLeNet_Weights.DEFAULT) # не умеет создаваться с None
        #print(model_googlenet)

        layers_dict = OrderedDict()
        # Проходим по слоям и сохраняем их имена1
        for name, layer in list(model_googlenet.named_children())[:-1]:
            layers_dict[name] = layer

        # Создаем nn.Sequential из этого OrderedDict
        self.features = nn.Sequential(layers_dict)

        #print(self.features)
        # изменение входа классификатора
        output_channels = self.features.conv1.conv.out_channels
        self.features.conv1.conv=nn.Conv2d(ipt_size[2], output_channels,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #print(self.features)

        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))    
        test_out =  self.features(test_ipt)

        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Linear(self.n_features, num_classes)
        self._init_classifier_weights()
        
        self.sigmoid = nn.Sigmoid() if use_end_sigmoid else None
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x if self.sigmoid is None else self.sigmoid(x)
        return x

    def _init_classifier_weights(self):
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.normal_(0, 0.01)
            self.classifier.bias.data.zero_()

    def block_requires_grad_features(self):
        # блокировка обучения сборщика фич
        for param in self.features.parameters():
            param.requires_grad = False

        # из-за модификации инпута необходимо его перетренировать
        for param in self.features.conv1.parameters():
            param.requires_grad = True


'''
# инициализация готовыми моделями
model_googlenet = Custom_Googlenet(ipt_size=(240, 240, 4), pretrained=True)
model_googlenet.to(device)
'''


TRANSFORMER_TYPES = {'swin_t'   : torchvision.models.swin_t, 
                     'swin_s'   : torchvision.models.swin_s,
                     'swin_b'   : torchvision.models.swin_b,
                     'swin_v2_t': torchvision.models.swin_v2_t,
                     'swin_v2_s': torchvision.models.swin_v2_s,
                     'swin_v2_b': torchvision.models.swin_v2_b,
                     'vit_b_16' : torchvision.models.vit_b_16, 
                     'vit_b_32' : torchvision.models.vit_b_32,
                     'vit_l_16' : torchvision.models.vit_l_16,
                     'vit_l_32' : torchvision.models.vit_l_32,
                     'vit_h_14' : torchvision.models.vit_h_14
                    }

TRANSFORMER_WEIGHTS = {'swin_t'   : torchvision.models.Swin_T_Weights, 
                       'swin_s'   : torchvision.models.Swin_S_Weights,
                       'swin_b'   : torchvision.models.Swin_B_Weights,
                       'swin_v2_t': torchvision.models.Swin_V2_T_Weights,
                       'swin_v2_s': torchvision.models.Swin_V2_S_Weights,
                       'swin_v2_b': torchvision.models.Swin_V2_B_Weights,
                       'vit_b_16' : torchvision.models.ViT_B_16_Weights, 
                       'vit_b_32' : torchvision.models.ViT_B_32_Weights,
                       'vit_l_16' : torchvision.models.ViT_L_16_Weights,
                       'vit_l_32' : torchvision.models.ViT_L_32_Weights,
                       'vit_h_14' : torchvision.models.ViT_H_14_Weights
                    }

class Custom_Transformer(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128, 3),
                 pretrained=True, 
                 transformer_type='swin_t', 
                 num_classes=1,
                 use_end_sigmoid=False):
        super(Custom_Transformer, self).__init__()
        
        
        assert transformer_type in TRANSFORMER_TYPES, "Unknown transformer_type '{}'".format(transformer_type)
        
                # load convolutional part of resnet
        if pretrained:
            try:
                transformer_weights=TRANSFORMER_WEIGHTS[transformer_type].IMAGENET1K_SWAG_E2E_V1
            except Exception:
                try:
                    transformer_weights=TRANSFORMER_WEIGHTS[transformer_type].IMAGENET1K_SWAG_LINEAR_V1
                except Exception:
                    transformer_weights =TRANSFORMER_WEIGHTS[transformer_type].IMAGENET1K_V1
        else:
            transformer_weights = None

        # load convolutional part of efficientnet
        transformer_model = TRANSFORMER_TYPES[transformer_type](weights=transformer_weights)

        #print(transformer_model)

        # изменение входа классификатора
        # Создаем nn.Sequential из этого OrderedDict
        self.features = nn.Sequential(transformer_model.features,
                                      transformer_model.norm,
                                      transformer_model.permute,
                                      transformer_model.avgpool,
                                      transformer_model.flatten)

        output_channels = self.features[0][0][0].out_channels
        self.features[0][0][0]=nn.Conv2d(ipt_size[2], output_channels, kernel_size=(4, 4), stride=(4, 4))
        
        #print(self.features)
        
        test_ipt = Variable(torch.zeros(1,ipt_size[2],ipt_size[0],ipt_size[1]))
        test_out = self.features(test_ipt)
        # изменение выхода классификатора
        self.head = nn.Linear(in_features=test_out.size(1), out_features=num_classes, bias=True)

        self._init_classifier_weights()

        self.sigmoid = nn.Sigmoid() if use_end_sigmoid else None

        #print("_"*20)
        #print(self.features)
    
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = x if self.sigmoid is None else self.sigmoid(x)

        return x

    def _init_classifier_weights(self):
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.normal_(0, 0.01)
            self.head.bias.data.zero_()

    def block_requires_grad_features(self):
        # блокировка обучения сборщика фич
        for param in self.features.parameters():
            param.requires_grad = False

        # из-за модификации инпута необходимо его перетренировать
        for param in self.features[0][0][0].parameters():
            param.requires_grad = True

'''
# инициализация готовыми моделями
model_transformer = Custom_Transformer(ipt_size=(240, 240, 4), pretrained=True)
model_transformer.to(device)
'''

def get_castom_model_by_name(name):
    if name.startswith("vgg"):
        return Custom_VGG
    elif name.startswith("resnet"):
        return Custom_ResNet
    elif name.startswith("efficientnet"):
        return Custom_EfficientNet
    elif name.startswith("densenet"):
        return Custom_DenseNet
    elif name == "googlenet":
        return Custom_Googlenet
    elif name.startswith("swin") or name.startswith("vit"):
        return Custom_Transformer