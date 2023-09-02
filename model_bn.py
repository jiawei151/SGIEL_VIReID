import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet18, resnet50, resnet101
from loss import sce, OriTripletLoss, shape_cpmt_cross_modal_ce
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        self.Wbn_shape = nn.BatchNorm2d(self.in_channels)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        nn.init.constant_(self.Wbn_shape.weight, 0.0)
        nn.init.constant_(self.Wbn_shape.bias, 0.0)


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x, shape=False):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        if shape:
            W_y = self.Wbn_shape(self.W[0](y))
        else:
            W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, isshape, modalbn):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1, isshape=isshape, onlyshallow=True, modalbn=modalbn)
        print('visible module:', model_v.isshape, model_v.modalbn)

        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x, modal=0):
        x = self.visible.conv1(x)
        if modal == 0: # RGB
            bbn1 = self.visible.bn1
        elif modal == 3: # shape
            bbn1 = self.visible.bn1_shape
        x = bbn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, isshape, modalbn):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1,isshape=isshape,onlyshallow=True, modalbn=modalbn)
        print('thermal resnet:', model_t.isshape, model_t.modalbn)

        # avg pooling to global pooling
        self.thermal = model_t


    def forward(self, x, modal=1):
        x = self.thermal.conv1(x)
        if modal == 1: # IR
            bbn1 = self.thermal.bn1
        elif modal == 3: # shape
            bbn1 = self.thermal.bn1_shape
        x = bbn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x




class base_resnet(nn.Module):
    def __init__(self, isshape, modalbn):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1, isshape=isshape, modalbn=modalbn)
        print('base resnet:', model_base.isshape, model_base.modalbn)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x, modal=0):
        x = self.base.layer1(x, modal)
        x = self.base.layer2(x, modal)
        x = self.base.layer3(x, modal)
        x = self.base.layer4(x, modal)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()
        self.isshape = True
        self.modalbn = 2

        self.thermal_module = thermal_module(self.isshape, 1)
        self.visible_module = visible_module(self.isshape, 1)
        self.base_resnet = base_resnet(self.isshape, self.modalbn)
        
        # TODO init_bn or not
        self.base_resnet.base.init_bn()
        self.thermal_module.thermal.init_bn()
        self.visible_module.visible.init_bn()
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        kk = 4
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


        if self.isshape:
            self.bottleneck_shape = nn.BatchNorm1d(pool_dim)
            self.bottleneck_shape.bias.requires_grad_(False)  # no shift
            self.classifier_shape = nn.Linear(pool_dim//kk, class_num, bias=False)

            self.projs = nn.ParameterList([])
            proj = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))
            # proj2 = nn.Parameter(torch.zeros([pool_dim,pool_dim//4*3], dtype=torch.float32, requires_grad=True))
            proj_shape = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))

            nn.init.kaiming_normal_(proj, nonlinearity="linear")        
            nn.init.kaiming_normal_(proj_shape, nonlinearity="linear")        
            self.bottleneck_shape.apply(weights_init_kaiming)
            self.classifier_shape.apply(weights_init_classifier)
            self.projs.append(proj)
            self.projs.append(proj_shape)
        if self.modalbn >= 2:
            self.bottleneck_ir = nn.BatchNorm1d(pool_dim)
            self.bottleneck_ir.bias.requires_grad_(False)  # no shift
            self.classifier_ir = nn.Linear(pool_dim//4, class_num, bias=False)
            self.bottleneck_ir.apply(weights_init_kaiming)
            self.classifier_ir.apply(weights_init_classifier)
        if self.modalbn == 3:
            self.bottleneck_modalx = nn.BatchNorm1d(pool_dim)
            self.bottleneck_modalx.bias.requires_grad_(False)  # no shift
            # self.classifier_ = nn.Linear(pool_dim, class_num, bias=False)
            self.bottleneck_modalx.apply(weights_init_kaiming)
            # self.classifier_rgb.apply(weights_init_classifier)
            self.proj_modalx = nn.Linear(pool_dim, pool_dim//kk, bias=False)
            self.proj_modalx.apply(weights_init_kaiming)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool
    def forward(self, x1, x2, x1_shape=None, x2_shape=None, mode=0):
        if mode == 0: # training

            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            if x1_shape is not None:
                x1_shape = self.visible_module(x1_shape, modal=3)
                x2_shape = self.thermal_module(x2_shape, modal=3)
                x_shape = torch.cat((x1_shape, x2_shape), 0)
            
        elif mode == 1: # eval rgb
            x = self.visible_module(x1)
        elif mode == 2: # eval ir
            x = self.thermal_module(x2)

        # shared block
        if mode > 0: # eval, only one modality per forward
            x = self.base_resnet(x, modal=mode-1)
        else: # training
            x1 = self.base_resnet(x1, modal=0)
            x2 = self.base_resnet(x2, modal=1)
            x = torch.cat((x1, x2), 0)
            
        if mode == 0 and x1_shape is not None: # shape for training
            x_shape = self.base_resnet(x_shape, modal=3)

        # gempooling
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        p = 3.0
        x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)

        if mode == 0 and x1_shape is not None:
            b, c, h, w = x_shape.shape
            x_shape = x_shape.view(b, c, -1)
            p = 3.0
            x_pool_shape = (torch.mean(x_shape**p, dim=-1) + 1e-12)**(1/p)

        # BNNeck
        if mode == 1:
            feat = self.bottleneck(x_pool)
        elif mode == 2:
            feat = self.bottleneck_ir(x_pool)
        elif mode == 0:
            assert x1.shape[0] == x2.shape[0]
            feat1 = self.bottleneck(x_pool[:x1.shape[0]])
            feat2 = self.bottleneck_ir(x_pool[x1.shape[0]:])
            feat = torch.cat((feat1, feat2), 0)
        if mode == 0 and x1_shape is not None:
            feat_shape = self.bottleneck_shape(x_pool_shape)

        # shape-erased feature
        if mode == 0:
            if x1_shape is not None:
                feat_p = torch.mm(feat, self.projs[0])
                proj_norm = F.normalize(self.projs[0], 2, 0) 
                
                feat_pnpn = torch.mm(torch.mm(feat, proj_norm), proj_norm.t())

                feat_shape_p = torch.mm(feat_shape, self.projs[1])

                logit2_rgbir = self.classifier(feat-feat_pnpn)
                logit_rgbir = self.classifier(feat)
                logit_shape = self.classifier_shape(feat_shape_p)

                return {'rgbir':{'bef':x_pool, 'aft':feat, 'logit': logit_rgbir, 'logit2': logit2_rgbir,'zp':feat_p,'other':feat-feat_pnpn},'shape':{'bef':x_pool_shape, 'aft':feat_shape, 'logit':logit_shape,'zp': feat_shape_p} }

            else:
                return x_pool, self.classifier(feat)
        else:

            return self.l2norm(x_pool), self.l2norm(feat)
    def myparameters(self):
        res = []
        for k, v in self.named_parameters():
            if v.requires_grad:
                if 'classifier' in k or 'proj' in k or 'bn' in k or 'bottleneck' in k:
                    continue
                res.append(v)
        return res
