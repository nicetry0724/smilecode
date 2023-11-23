# -*- coding: utf-8 -*-
# create time:2022/9/28
# author:Pengze Li
import torch
import torch.nn as nn
from torch.autograd import Variable
from LeNet3d import LeNet3D
import warnings
from PIL import Image
import torchvision.transforms as Tr

warnings.filterwarnings('ignore')


new_parameter = {}
parameter = torch.load("pre-trained model address")
for k, v in parameter.items():
    if 'features' in k:
        name = k[9:]
        # print(name)
        new_parameter[name] = v





def LN_loss(feature_module,loss_func1,loss_func2,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss1=loss_func1(out,out_)
    loss2=loss_func2(out,out_)
    return loss1, loss2, out, out_


def get_feature_module(layer_index,device=None):
    LN = LeNet3D().features
    LN.load_state_dict(new_parameter)
    for parm in LN.parameters():
        parm.requires_grad = False

    feature_module = LN[layer_index]
    feature_module.to(device)
    return feature_module


class PerceptualLoss(nn.Module):
    def __init__(self,loss_func1,loss_func2,layer_indexs=None,device=None):
        super(PerceptualLoss, self).__init__()
        self.creation1=loss_func1
        self.creation2=loss_func2
        self.layer_indexs=layer_indexs
        self.device=device

    def forward(self,y,y_):
        loss_1 = 0
        for index in self.layer_indexs:
            feature_module=get_feature_module(index,self.device)
            loss1, loss2, y, y_ = LN_loss(feature_module,self.creation1,self.creation2,y,y_)
            loss_1 += loss1 + 0.1 * loss2
        return loss_1


def PerceptualLoss_MSE(y_true,y_pred):
    x = y_true
    y = y_pred
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_indexs = [0,4,8]
    loss_func1 = nn.MSELoss().to(device)
    loss_func2 = nn.L1Loss().to(device)
    creation = PerceptualLoss(loss_func1, loss_func2, layer_indexs, device)
    perceptual_loss=creation(x,y)
    return perceptual_loss
