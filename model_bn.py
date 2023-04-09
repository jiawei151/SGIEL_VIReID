import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet18, resnet50
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
        elif modal == 2: # modalx
            bbn1 = self.visible.bn1_modalx
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
        elif modal == 2: # modalx
            bbn1 = self.thermal.bn1_modalx
        elif modal == 3: # shape
            bbn1 = self.thermal.bn1_shape
        x = bbn1(x)

        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class visible_module_twostream(nn.Module):
    def __init__(self, isshape, modalbn):
        super(visible_module_twostream, self).__init__()

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

        x1 = self.visible.layer1(x, modal)
        x2 = self.visible.layer2(x1[0], modal)
        x3 = self.visible.layer3(x2[0], modal)
        x4 = self.visible.layer4(x3[0], modal)
        # return x1,x2,x3,x4
        return x4[0]


class thermal_module_twostream(nn.Module):
    def __init__(self, isshape, modalbn):
        super(thermal_module_twostream, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1,isshape=isshape,onlyshallow=True, modalbn=modalbn)
        print('thermal resnet:', model_t.isshape, model_t.modalbn)

        # avg pooling to global pooling
        self.thermal = model_t


    def forward(self, x, modal=0):
        x = self.thermal.conv1(x)
        if modal == 0: # IR
            bbn1 = self.thermal.bn1
        elif modal == 3: # shape
            bbn1 = self.thermal.bn1_shape
        x = bbn1(x)

        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x1 = self.thermal.layer1(x, modal)
        x2 = self.thermal.layer2(x1[0], modal)
        x3 = self.thermal.layer3(x2[0], modal)
        x4 = self.thermal.layer4(x3[0], modal)

        return x4[0]



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
        x1 = self.base.layer1(x, modal)
        x2 = self.base.layer2(x1[0], modal)
        x3 = self.base.layer3(x2[0], modal)
        x4 = self.base.layer4(x3[0], modal)
        return x1,x2,x3,x4

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-12, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.pow(self.p)
        # x = x.clamp(min=self.eps).pow(self.p)
        return (F.adaptive_avg_pool2d(x, self.output_size)+self.eps).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
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
        self.thermal_module.thermal.init_bn(onlyshallow=True)
        self.visible_module.visible.init_bn(onlyshallow=True)
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            # self.NL_1_shape = nn.ModuleList(
            #     [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            # self.NL_2_shape = nn.ModuleList(
            #     [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            # self.NL_3_shape = nn.ModuleList(
            #     [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            # self.NL_4_shape = nn.ModuleList(
            #     [Non_local(2048) for i in range(non_layers[3])])
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
            # self.classifier3 = nn.Linear(1024, class_num, bias=False)
            # self.classifier_shape3 = nn.Linear(512, class_num, bias=False)

            self.projs = nn.ParameterList([])
            proj = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))
            # proj2 = nn.Parameter(torch.zeros([pool_dim,pool_dim//4*3], dtype=torch.float32, requires_grad=True))
            proj_shape = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))

            # proj2 = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))

            # self.proj = nn.Linear(pool_dim, pool_dim//kk, bias=False)
            # self.proj_shape = nn.Linear(pool_dim,pool_dim//kk, bias=False)
            
            # self.proj3 = nn.Linear(1024, 512, bias=False)
            # self.proj3_shape = nn.Linear(1024, 512, bias=False)

            
            # self.proj_local_shape = nn.Linear(pool_dim,pool_dim//kk, bias=False)
            # self.proj.apply(weights_init_kaiming)
            # self.proj3.apply(weights_init_kaiming)
            # self.proj_local.apply(weights_init_kaiming)
            # self.proj_shape.apply(weights_init_kaiming)
            # self.proj3_shape.apply(weights_init_kaiming)
            
            # self.proj_local_shape.apply(weights_init_kaiming)

            nn.init.kaiming_normal_(proj, nonlinearity="linear")        
            # nn.init.kaiming_normal_(proj2, nonlinearity="linear")        
            nn.init.kaiming_normal_(proj_shape, nonlinearity="linear")        
            self.bottleneck_shape.apply(weights_init_kaiming)
            self.classifier_shape.apply(weights_init_classifier)
            self.projs.append(proj)
            self.projs.append(proj_shape)
            # self.projs.append(proj2)
            # self.classifier3.apply(weights_init_classifier)
            # self.classifier_shape3.apply(weights_init_classifier)
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
        # self.loss_tri = OriTripletLoss(margin=0.1)
        # self.my_gem_pool = GeneralizedMeanPooling(output_size=(6,1))
        # self.featmap = []
        # self.transformer = getTransformer(class_num)


    # def average_bnneck(self):
    #         tmpw = self.bottleneck.weight.data.clone() + self.bottleneck_ir.weight.data.clone() + self.bottleneck_shape.weight.data.clone()# + self.bottleneck_modalx.weight.data.clone()
    #         tmpw /= 3
    #         self.bottleneck.weight.data = tmpw.clone()

    #         if self.isshape:
    #             self.bottleneck_shape.weight.data = tmpw.clone()
    #         if self.modalbn >= 2:
    #             self.bottleneck_ir.weight.data = tmpw.clone()
    #         if self.modalbn == 3:
    #             self.bottleneck_modalx.weight.data = tmpw.clone()

    # def hook(self, module, inputs, output):
    #     self.featmap.append(inputs)
    def mybatchnorm1d(self, x, bn):
        # x = x.permute(0,2,1)
        # x_view = x.reshape(-1, x.shape[2])
        x_mean = torch.mean(x, dim=0).detach()
        x_var = torch.var(x, dim=0).detach()
        x_hat = (x - x_mean) / (torch.sqrt(x_var  + 1e-5))
        out = bn.weight.data * x_hat + bn.bias.data
        return out
    def forward(self, x1, x2, x1_shape=None, x2_shape=None, mode=0, y=None):
        if mode == 0: # training
            # x1_modalx = self.visible_module(x1, modal=2)
            # x2_modalx = self.thermal_module(x2, modal=2)

            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            if x1_shape is not None:
                x1_shape = self.visible_module(x1_shape, modal=3)
                x2_shape = self.thermal_module(x2_shape, modal=3)
                x_shape = torch.cat((x1_shape, x2_shape), 0)
            
            # x_modalx = torch.cat((x1_modalx, x2_modalx), 0)
            nextmodal = 2
        elif mode == 1: # eval rgb
            if x1_shape is not None:
                x = self.visible_module(x1, modal=3)
            else:
                x = self.visible_module(x1)
            nextmodal = 0
        elif mode == 2: # eval ir
            if x1_shape is not None:
                x = self.thermal_module(x2, modal=3)
            else:
                x = self.thermal_module(x2)
            nextmodal = 1

        # shared block
        if mode > 0:
            if x2_shape is not None:
                x, _ = self.base_resnet(x, modal=3)[3]
            else:
                x, _ = self.base_resnet(x, modal=nextmodal)[3]
            
        else:
            # handle = self.base_resnet.base.layer4[2].relu.register_forward_hook(self.hook)
            x1l1, x1l2, x1l3, x1l4 = self.base_resnet(x1, modal=0)
            x1, x1_bef = x1l4
            x2l1, x2l2, x2l3, x2l4 = self.base_resnet(x2, modal=1)
            x2, x2_bef = x2l4
            x = torch.cat((x1, x2), 0)
            x_featmap3 = torch.cat((x1l3[0], x2l3[0]), 0)
            # x_bef = torch.cat((x1_bef, x2_bef), 0)
            
        if mode == 0 and x1_shape is not None:
            x_shapel1, x_shapel2, x_shapel3, x_shapel4 = self.base_resnet(x_shape, modal=3)
            x_shape, _ = x_shapel4 
            # x_shape, _ = x_shapel2
            # handle.remove()

        #### split featmap ####
        # x_split = self.my_gem_pool(x)
        # if x1_shape is not None:
        #     x_shape_split = self.my_gem_pool(x_shape)

        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)

            if mode == 0 and x1_shape is not None:
                b, c, h, w = x_shape.shape
                x_shape = x_shape.view(b, c, -1)
                p = 3.0
                x_pool_shape = (torch.mean(x_shape**p, dim=-1) + 1e-12)**(1/p)

                # b, c, h, w = x_featmap3.shape
                # x_featmap3 = x_featmap3.view(b, c, -1)
                # x_shape_featmap3 = x_shape_featmap3.view(b, c, -1)
                # p = 3.0
                # x_pool_featmap3 = (torch.mean(x_featmap3**p, dim=-1) + 1e-12)**(1/p)
                # x_pool_shape_featmap3 = (torch.mean(x_featmap3**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        if mode > 0:
            if nextmodal == 0:
                feat = self.bottleneck(x_pool)
            elif x1_shape is not None:
                feat = self.bottleneck_shape(x_pool)
            else:
                feat = self.bottleneck_ir(x_pool)
        else:
            assert x1.shape[0] == x2.shape[0]
            feat1 = self.bottleneck(x_pool[:x1.shape[0]])
            feat2 = self.bottleneck_ir(x_pool[x1.shape[0]:])
            feat = torch.cat((feat1, feat2), 0)
        # feat = self.bottleneck(x_pool)
        if mode == 0 and x1_shape is not None:
            feat_shape = self.bottleneck_shape(x_pool_shape)

        if mode == 0:
            # if self.training:
            if x1_shape is not None:
                #### channel-level ####
                feat_p = torch.mm(feat, self.projs[0])
                proj_norm = F.normalize(self.projs[0], 2, 0) 
                # feat_pother = torch.mm(feat, self.projs[2])
                
                feat_pnpn = torch.mm(torch.mm(feat, proj_norm), proj_norm.t())

                feat_shape_p = torch.mm(feat_shape, self.projs[1])

                logit2_rgbir = self.classifier(feat-feat_pnpn)
                # logit2_rgbir = self.classifier_ir(feat_pother)
                logit_rgbir = self.classifier(feat)
                logit_shape = self.classifier_shape(feat_shape_p)

                # logit3_rgbir = self.classifier(feat-feat_pnpn+torch.cat((feat_pnpn[x1.shape[0]:],feat_pnpn[:x1.shape[0]]),0))

                

                # feat_shape_hat = torch.mm(F.relu(feat_p), self.projs[1])
                # loss_mse = ((feat_p - feat_shape_p.detach()) ** 2).mean(1).mean()

                # loss_kl = sce(torch.mm(feat_p, self.classifier_shape.weight.data.detach().t()), logit_shape)
                # # loss_kl = sce(self.classifier_shape(feat_p), self.classifier_shape(feat_shape_p))
                # if loss_mse > 1e6 or loss_kl > 1e6:
                #     import pdb
                #     pdb.set_trace()
                
                #### multi-scale head ####
                # x_pool_featmap3_p = self.proj3(x_pool_featmap3)
                # proj_norm = F.normalize(self.proj3.weight.t(), 2, 0) 
                # feat_pnpn = torch.mm(torch.mm(x_pool_featmap3, proj_norm), proj_norm.t())
                # logit2_rgbir3 = self.classifier3(x_pool_featmap3-feat_pnpn)
                # logit_rgbir3 = self.classifier3(x_pool_featmap3)

                # x_pool_shape_featmap3_p = self.proj3_shape(x_pool_shape_featmap3)
                # logit_shape3 = self.classifier_shape3(x_pool_shape_featmap3_p)

                # # loss_mse3 = ((x_pool_featmap3_p - x_pool_shape_featmap3_p.detach()) ** 2).mean(1).mean()

                # loss_kl3 = sce(torch.mm(x_pool_featmap3_p, self.classifier_shape3.weight.data.detach().t()), logit_shape3)
                # # loss_kl = sce(self.classifier_shape(feat_p), self.classifier_shape(feat_shape_p))
                # if loss_kl3 > 1e6:
                #     import pdb
                #     pdb.set_trace()
                #### coarse2fine ####
                # logit_x_shape_c2f, x_shape_c2f = self.transformer(x_shapel1[0], x_shapel2[0], x_shapel3[0], x_shapel4[0])
                # loss_kl = sce(torch.mm(feat_p, self.classifier_shape.weight.data.detach().t()), logit_shape)
                # loss_kl_c2f = sce(torch.mm(feat_p, self.transformer.proj.weight.data.detach().t()), logit_x_shape_c2f)
                
                #### spatial-level ####
                # x = torch.cat((self.mybatchnorm1d(x[:x1.shape[0]], self.bottleneck), self.mybatchnorm1d(x[x1.shape[0]:], self.bottleneck_ir)),0)
                # x_shape = self.mybatchnorm1d(x_shape, self.bottleneck_shape)
                # x_shape = x_shape_bef.view(b, c, -1)
                # x = x_bef.view(b, c, -1)
                # x = x.permute(0,2,1)
                # x_shape = x_shape.permute(0,2,1)
                # x = self.proj_local(x)
                # x_shape = self.proj_shape(x_shape)

                # x_p = self.proj(x) # 64 162 512
                # x_p = torch.einsum('blc,cd->bld', x,self.proj.weight.data.detach().t()) # 64 162 512
                # x_p = x.permute(0,2,1) # 64 162 512
                # x_norm = F.normalize(x, 2, -1)
                # x_shape_norm = F.normalize(x_shape, 2, -1)
                # x_p_norm = F.normalize(x_p, 2, -1)
                # x_shape_p = self.proj_shape(x_shape)
                # x_shape_p = torch.einsum('blc,cd->bld', x_shape, self.proj_shape.weight.data.detach().t())
                # x_shape_p_norm = F.normalize(x_shape_p, 2, -1)

                # x_selfcos = torch.bmm(x_norm, x_norm.permute(0,2,1))
                # x_p_selfcos = torch.bmm(x_p_norm, x_p_norm.permute(0,2,1))
                # x_shape_p_selfcos = torch.bmm(x_shape_p_norm, x_shape_p_norm.permute(0,2,1))
                # x_shape_selfcos = torch.bmm(x_shape_norm, x_shape_norm.permute(0,2,1))
                # loss_mse_relation = ((x_selfcos-x_shape_selfcos.detach())**2).mean()
                # loss_mse_gl_shape = (((torch.mean(x_shape**3, dim=1) + 1e-12)**(1/3)-feat_shape_p.detach())**2).mean()
                # loss_mse_gl = ((torch.mean(x, dim=1)-feat_p.detach())**2).mean()
                # x_pn = torch.einsum('blc,cd->bld',x,proj_norm)
                # x_pnpn = torch.einsum('bld,dc->blc',x_pn,proj_norm.t())
                # x_pnpn = torch.mm(torch.mm(x.reshape(-1,x.shape[2]), proj_norm), proj_norm.t())
                # x_pnpn = x_pnpn.reshape(b, -1, c)
                # x_other = x - x_pnpn

                # loss_triplet = self.loss_tri(x_other.reshape(b, -1), y)[0]/10
                # loss_triplet = []
                # for i in range(6):
                #     loss_triplet.append(self.loss_tri(x[:,i,:], y)[0])
                # loss_triplet = sum(loss_triplet)/len(loss_triplet)

                # logit_x_other = torch.einsum('blc,cd->bld',x_other, self.classifier.weight.data.t().detach())

                # logit_x_other = self.classifier(x_other)
                # loss_kl_featmap = []
                # for i in range(6):
                #     loss_kl_featmap.append(sce(logit_x_other[:,i,:][:x1.shape[0]], logit_x_other[:,i,:][x1.shape[0]:])+sce(logit_x_other[:,i,:][x1.shape[0]:], logit_x_other[:,i,:][:x1.shape[0]]))
                # loss_kl_featmap = sum(loss_kl_featmap)/len(loss_kl_featmap)


                # x_other_norm = F.normalize(x_other, 2, -1)
                # x_pool_norm = F.normalize(x_pool, 2, -1)
                # x_other_glcos = torch.einsum('blc,bc->bl', x_other_norm, x_pool_norm)
                # x_glcos = torch.einsum('blc,bc->bl', x_norm, x_pool_norm)
                # loss_mse_gl = ((x_other_glcos-x_glcos.detach())**2).mean()


                # modal_shape_cos_max = modal_shape_cos.max(dim=2,keepdim=True)[0]
                # selected_x_pool = x.permute(0,2,1)[(modal_shape_cos_max==modal_shape_cos).squeeze()]
                # selected_x_shape_pool = x_shape.permute(0,2,1)[(modal_shape_cos_max==modal_shape_cos).squeeze()]
                # # selected_x = torch.cat((self.bottleneck(selected_x_pool[:x1.shape[0]]), self.bottleneck_ir(selected_x_pool[x1.shape[0]:])),0)
                # # selected_x_shape = self.bottleneck_shape(selected_x_shape_pool) 
                # selected_x = torch.cat((self.mybatchnorm1d(selected_x_pool[:x1.shape[0]], self.bottleneck), self.mybatchnorm1d(selected_x_pool[x1.shape[0]:],self.bottleneck_ir)),0)
                # selected_x_shape = self.mybatchnorm1d(selected_x_shape_pool, self.bottleneck_shape) 

                # selected_x_p = self.proj(selected_x)
                # selected_x_pnpn = torch.mm(torch.mm(selected_x, proj_norm), proj_norm.t())
                # logit2_selected = self.classifier(selected_x-selected_x_pnpn)
                # # logit_selected = self.classifier(selected_x)

                # selected_x_shape_p = self.proj_shape(selected_x_shape)
                # logit_selected_shape = self.classifier_shape(feat_shape_p)
                # loss_mse_selected = ((selected_x_p - selected_x_shape_p.detach()) ** 2).mean(1).mean()
                # if not selected_x_p.shape[0] == logit_selected_shape.shape[0]:
                #     import pdb
                #     pdb.set_trace()
                # loss_kl_selected = sce(torch.mm(selected_x_p, self.classifier_shape.weight.data.detach().t()), logit_selected_shape)
                
                # shape_cam = x_shape.mean(1, keepdim=True) 
                # shape_featmap = x*(modal_shape_cos>modal_shape_cos.mean()).detach()*(shape_cam>shape_cam.mean()).detach()
                # other_featmap = x-shape_featmap
                # p = 3.0 # TODO: may be higher?
                # shapex_pool = (torch.mean(shape_featmap**p, dim=-1) + 1e-12)**(1/p)
                # otherx_pool = (torch.mean(other_featmap**p, dim=-1) + 1e-12)**(1/p)

                # TODO: eval()?

                # shapefeat1 = self.mybatchnorm1d(shapex_pool[:x1.shape[0]], self.bottleneck)
                # shapefeat2 = self.mybatchnorm1d(shapex_pool[x1.shape[0]:], self.bottleneck_ir)
                # # shapefeat1 = self.bottleneck(shapex_pool[:x1.shape[0]])
                # # shapefeat2 = self.bottleneck_ir(shapex_pool[x1.shape[0]:])
                # shapefeat = torch.cat((shapefeat1, shapefeat2), 0)
                # otherfeat1 = self.mybatchnorm1d(otherx_pool[:x1.shape[0]], self.bottleneck)
                # otherfeat2 = self.mybatchnorm1d(otherx_pool[x1.shape[0]:], self.bottleneck_ir)
                # # otherfeat1 = self.bottleneck(otherx_pool[:x1.shape[0]])
                # # otherfeat2 = self.bottleneck_ir(otherx_pool[x1.shape[0]:])
                # otherfeat = torch.cat((otherfeat1, otherfeat2), 0)
                # shapefeat = shapex_pool
                # otherfeat = otherx_pool

                # TODO: add contraint on CAMs or not, or how to?
                # loss_cam = (((x.mean(1,keepdim=True)-x_shape.mean(1,keepdim=True))**2)[(modal_shape_cos>modal_shape_cos.mean())*(shape_cam>shape_cam.mean())]).sum()/x.shape[0]

                # loss_ce = F.cross_entropy(self.classifier(shapefeat), y)+F.cross_entropy(self.classifier(otherfeat), y)




                # loss_kl2 = sce(self.classifier_shape(feat_shape_pp), self.classifier_shape(feat_pp))

                # shape_normed = F.normalize(feat_shape_p, p=2, dim=1)
                # shape_cossim = torch.mm(shape_normed,shape_normed.t())
                # batch_size = y.shape[0]
                # mask = y.expand(batch_size,batch_size).eq(y.expand(batch_size, batch_size).t())
                # idx_temp = torch.arange(batch_size)
                # # mask = idx_temp.expand(batch_size,batch_size).eq(idx_temp.expand(batch_size, batch_size).t())
                # target4modal = []
                # for i in range(batch_size):
                #     mask[i][i] = False
                #     target4modal.append(idx_temp[mask[i]][shape_cossim[i][mask[i]].argmax()].unsqueeze(0))
                # target4modal = torch.cat(target4modal)
                # loss_kl2 = sce(self.classifier_shape(feat_p),self.classifier_shape(feat_shape_p[target4modal]))



                # loss_ce = F.cross_entropy(torch.mm(feat_p, self.classifier_shape.weight.data.detach().t()), y)
                # loss_sp = feat_distill_loss(feat_p, feat_shape_p)
                # return {'rgbir':{'bef':x_pool, 'aft':feat, 'logit': self.classifier(feat), 'logit2': self.classifier(feat-feat_pnpn),'s-ortho':feat-feat_pnpn},'shape':{'bef':x_pool_shape, 'aft':feat_shape, 'logit':self.classifier_shape(feat_shape), 'logit2':torch.mm(feat_shape_pp, self.classifier_shape.weight.data.detach().t())}}, loss_kl+loss_mse
                # return {'rgbir':{'bef':x_pool, 'aft':feat, 'logit': self.classifier(feat), 'logit2': self.classifier(feat-feat_pnpn),'s-ortho':feat-feat_pnpn},'shape':{'logit':self.classifier_shape(feat_shape_p),'zp': feat_shape_p }}, loss_kl+loss_mse
                return {'rgbir':{'bef':x_pool, 'aft':feat, 'logit': logit_rgbir, 'logit2': logit2_rgbir,'zp':feat_p,'other':feat-feat_pnpn},'shape':{'bef':x_pool_shape, 'aft':feat_shape, 'logit':logit_shape,'zp': feat_shape_p} }
                # return {'rgbir':{'bef':x_pool, 'aft':feat, 'logit': logit_rgbir, 'logit2': logit2_rgbir, 'logit3': logit_rgbir3, 'logit23': logit2_rgbir3},'shape':{'aft':feat_shape, 'logit':logit_shape, 'logit3':logit_shape3, 'zp': feat_shape_p, 'zp3': x_pool_shape_featmap3_p} }, loss_kl+loss_mse+loss_kl3

            else:
                return x_pool, self.classifier(feat)
        else:
            # feat = self.proj(feat)
            # alpha = self.alpha * 0.01 + -1
            # proj_norm =  F.normalize(self.proj.weight.t(), 2, 0) 
            # feat_pnpn = torch.mm(torch.mm(feat, proj_norm), proj_norm.t())
            # feat = (1.-alpha)*(feat - feat_pnpn) + (1.+alpha)*feat_pnpn
            # feat_p = torch.mm(feat, self.projs[0])
            # proj_norm = F.normalize(self.projs[0], 2, 0) 
                # feat_pother = torch.mm(feat, self.projs[2])
                
            # feat_pnpn = torch.mm(torch.mm(feat, proj_norm), proj_norm.t())

            return self.l2norm(x_pool), self.l2norm(feat)#, self.l2norm(feat_p), self.l2norm(feat-feat_pnpn)
    def myparameters(self):
        res = []
        for k, v in self.named_parameters():
            if v.requires_grad:
                if 'classifier' in k or 'proj' in k or 'bn' in k or 'bottleneck' in k:
                    continue
                res.append(v)
        return res
    def get_loss_newfeat(self, oldfeat, newfeat, feat_shape, y):
        newfeat = newfeat.detach()
        feat_shape = feat_shape.detach()
        feat_shape_p = self.proj_shape(feat_shape)
        #### channel-level ####
        # newfeat1 = self.bottleneck(newfeat[:newfeat.shape[0]//2])
        # newfeat2 = self.bottleneck_ir(newfeat[newfeat.shape[0]//2:])
        # newfeat1 = self.mybatchnorm1d(newfeat[:newfeat.shape[0]//2], self.bottleneck)
        # newfeat2 = self.mybatchnorm1d(newfeat[newfeat.shape[0]//2:], self.bottleneck_ir)
        # newfeat = torch.cat((newfeat1, newfeat2), 0)
        feat_p = self.proj(newfeat)
        proj_norm = F.normalize(self.proj.weight.t(), 2, 0) 
        feat_pnpn = torch.mm(torch.mm(newfeat, proj_norm), proj_norm.t())
        # logit2_rgbir = self.classifier(newfeat-feat_pnpn)
        logit2_rgbir = torch.mm(newfeat-feat_pnpn, self.classifier.weight.data.detach().t())
        # logit_shape = self.classifier_shape(feat_shape_p)
        logit_shape = torch.mm(feat_shape_p, self.classifier_shape.weight.data.detach().t())
        loss_ce = F.cross_entropy(logit2_rgbir, y)

        loss_mse = ((feat_p - feat_shape_p.detach()) ** 2).mean(1).mean()

        loss_kl = sce(torch.mm(feat_p, self.classifier_shape.weight.data.detach().t()), logit_shape)

        # loss_kl_rgbir = sce(logit_rgbir[:y.shape[0]//2],logit_rgbir[y.shape[0]//2:])+sce(logit_rgbir[y.shape[0]//2:],logit_rgbir[:y.shape[0]//2])
        
        loss_kl_rgbir2 = shape_cpmt_cross_modal_ce(y[:y.shape[0]//2], y[:y.shape[0]//2], {'rgbir':{'logit2': logit2_rgbir,},'shape':{'zp': feat_shape_p } })

        # loss_kl = sce(self.classifier_shape(feat_p), self.classifier_shape(feat_shape_p))
        if loss_mse > 1e6 or loss_kl > 1e6:
            import pdb
            pdb.set_trace()


        return loss_kl+loss_mse+loss_ce+loss_kl_rgbir2


class embed_net_twostream(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net_twostream, self).__init__()
        self.isshape = True
        self.modalbn = 1

        self.thermal_module = thermal_module_twostream(self.isshape, 1)
        self.visible_module = visible_module_twostream(self.isshape, 1)
        # self.base_resnet = base_resnet(self.isshape, self.modalbn)
        
        # TODO init_bn or not
        # self.base_resnet.base.init_bn()
        self.thermal_module.thermal.init_bn()
        self.visible_module.visible.init_bn()
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            # self.NL_1_shape = nn.ModuleList(
            #     [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            # self.NL_2_shape = nn.ModuleList(
            #     [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            # self.NL_3_shape = nn.ModuleList(
            #     [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            # self.NL_4_shape = nn.ModuleList(
            #     [Non_local(2048) for i in range(non_layers[3])])
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
            self.bottleneck_shape2 = nn.BatchNorm1d(pool_dim)
            self.bottleneck_shape2.bias.requires_grad_(False)  # no shift
            self.classifier_shape = nn.Linear(pool_dim//kk, class_num, bias=False)
            # self.classifier3 = nn.Linear(1024, class_num, bias=False)
            # self.classifier_shape3 = nn.Linear(512, class_num, bias=False)

            self.projs = nn.ParameterList([])
            proj = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))
            # proj2 = nn.Parameter(torch.zeros([pool_dim,pool_dim//4*3], dtype=torch.float32, requires_grad=True))
            proj_shape = nn.Parameter(torch.zeros([pool_dim,pool_dim//kk], dtype=torch.float32, requires_grad=True))


            nn.init.kaiming_normal_(proj, nonlinearity="linear")
            # nn.init.kaiming_normal_(proj2, nonlinearity="linear")        
            nn.init.kaiming_normal_(proj_shape, nonlinearity="linear")        
            self.bottleneck_shape.apply(weights_init_kaiming)
            self.bottleneck_shape2.apply(weights_init_kaiming)
            self.classifier_shape.apply(weights_init_classifier)
            self.projs.append(proj)
            self.projs.append(proj_shape)
            # self.projs.append(proj2)
            # self.classifier3.apply(weights_init_classifier)
            # self.classifier_shape3.apply(weights_init_classifier)

            self.bottleneck_ir = nn.BatchNorm1d(pool_dim)
            self.bottleneck_ir.bias.requires_grad_(False)  # no shift
            self.classifier_ir = nn.Linear(pool_dim//4, class_num, bias=False)
            self.bottleneck_ir.apply(weights_init_kaiming)
            self.classifier_ir.apply(weights_init_classifier)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool


    def forward(self, x1, x2, x1_shape=None, x2_shape=None, mode=0, y=None):
        if mode == 0: # training

            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1,x2),0)
            
            if x1_shape is not None:
                x1_shape = self.visible_module(x1_shape, modal=3)
                x2_shape = self.thermal_module(x2_shape, modal=3)
                x_shape = torch.cat((x1_shape, x2_shape), 0)

        elif mode == 1: # eval rgb
            x = self.visible_module(x1)

        elif mode == 2: # eval ir
            x = self.thermal_module(x2)

            
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)

            if mode == 0 and x1_shape is not None:
                b, c, h, w = x_shape.shape
                x_shape = x_shape.view(b, c, -1)
                p = 3.0
                x_pool_shape = (torch.mean(x_shape**p, dim=-1) + 1e-12)**(1/p)

        if mode > 0: #eval
            if mode == 1: # rgb
                feat = self.bottleneck(x_pool)
            elif mode == 2: # ir
                feat = self.bottleneck_ir(x_pool)
        else:
            assert x1.shape[0] == x2.shape[0]
            feat1 = self.bottleneck(x_pool[:x1.shape[0]])
            feat2 = self.bottleneck_ir(x_pool[x1.shape[0]:])
            feat = torch.cat((feat1, feat2), 0)
        if mode == 0 and x1_shape is not None:
            feat_shape1 = self.bottleneck_shape(x_pool_shape[:x1.shape[0]])
            feat_shape2 = self.bottleneck_shape2(x_pool_shape[x1.shape[0]:])
            feat_shape = torch.cat((feat_shape1,feat_shape2),0)

        if mode == 0:
            # if self.training:
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

            # feat_p = torch.mm(feat, self.projs[0])
            # proj_norm = F.normalize(self.projs[0], 2, 0) 
       
            # feat_pnpn = torch.mm(torch.mm(feat, proj_norm), proj_norm.t())

            return self.l2norm(x_pool), self.l2norm(feat)#, self.l2norm(feat_p), self.l2norm(feat-feat_pnpn)
