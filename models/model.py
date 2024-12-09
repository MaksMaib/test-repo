import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import torch.optim as optim


def CustomefficientnetV2M(num_classes, pretrained=True, fixed_feature_extr=True):

    if pretrained:
        model_ft = models.efficientnet_v2_m(weights="EfficientNet_V2_M_Weights.DEFAULT")
    else:
        model_ft = models.efficientnet_v2_m()

    model_ft.classifier[1] = nn.Linear(1280, num_classes)
    model_ft.num_classes = num_classes

    if fixed_feature_extr:
        optimizer = optim.SGD(model_ft.classifier.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    return model_ft, optimizer


def Resnet18(num_classes, pretrained=True, fixed_feature_extr=True):

    if pretrained:
        model_ft = models.efficientnet_v2_m(weights="EfficientNet_V2_M_Weights.DEFAULT")
    else:
        model_ft = models.efficientnet_v2_m()

    model_ft.classifier[1] = nn.Linear(1280, num_classes)
    model_ft.num_classes = num_classes

    if fixed_feature_extr:
        optimizer = optim.SGD(model_ft.classifier.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    return model_ft, optimizer


