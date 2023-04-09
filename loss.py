from turtle import position
from urllib.parse import quote_plus
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size=None, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct    
        
        # modal = (torch.arange(n) // (n/2)).cuda()
        # modalmask = modal.expand(n, n).ne(modal.expand(n, n).t())

# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)
        
        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow =torch.clamp_max(diff_pow, max=10)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            
            loss = self.ranking_loss(diff_pow, y)
        
        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

def sce(new_logits, old_logits):
    loss_ke_ce = (- F.softmax(old_logits, dim=1).detach() * F.log_softmax(new_logits,dim=1)).mean(0).sum()
    return loss_ke_ce

def reweight_sce(new_logits, old_logits, w):
    loss_ke_ce = ((- F.softmax(old_logits, dim=1).detach() * F.log_softmax(new_logits,dim=1)).sum(1)*w).mean()
    return loss_ke_ce

def sce_nodetach(new_logits, old_logits):
    loss_ke_ce = (- F.softmax(old_logits, dim=1) * F.log_softmax(new_logits,dim=1)).mean(0).sum()
    return loss_ke_ce
def exchange_sce(new_logits, old_logits, y):

    positions = torch.arange(new_logits.shape[0])
    assert (y[positions%2==0] == y[positions%2==1]).all()
    loss_ke_ce1 = (- F.softmax(old_logits[positions%2==1], dim=1).detach() * F.log_softmax(new_logits[positions%2==0],dim=1)).sum()
    loss_ke_ce2 = (- F.softmax(old_logits[positions%2==0], dim=1).detach() * F.log_softmax(new_logits[positions%2==1],dim=1)).sum()
    return (loss_ke_ce1+loss_ke_ce2)/ y.shape[0]

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    # input_softmax = input_logits
    target_softmax = F.softmax(target_logits, dim=1)
    # target_softmax = target_logits
    return F.mse_loss(input_softmax, target_softmax, reduction='sum')/ input_softmax.shape[0]

def feat_distill_loss(feat_1, feat_2):
    L1_loss = torch.nn.L1Loss()
    return L1_loss(cosine_similarity(feat_1,feat_1), cosine_similarity(feat_2.detach(),feat_2.detach()))

def cosine_similarity(fea1, feat2):
    input1_normed = F.normalize(fea1, p=2, dim=1)
    input2_normed = F.normalize(feat2, p=2, dim=1)
    distmat = torch.mm(input1_normed, input2_normed.t())
    return distmat

def similarity_preserve_loss(input, target, t=5):
    n = input.size(0)
    target = target
    input_view = F.normalize(input, 2, 1).view(n, -1)
    target_view = F.normalize(target, 2, 1).view(n, -1)

    target_view_inner = target_view.mm(target_view.t()).view(n, n)
    input_view_inner = input_view.mm(input_view.t())
    target_view_inner = target_view_inner
    # input_view_inner_nodiag = torch.zeros(n, n-1).cuda()
    # target_view_inner_nodiag = torch.zeros(n, n-1).cuda()

    # for i in range(n):
    #     ind = [ii for ii in range(n)]
    #     ind.pop(i)
    #     input_view_inner_nodiag[i] = input_view_inner[i][ind]
    #     target_view_inner_nodiag[i] = target_view_inner[i][ind]

    # x_logsm = F.log_softmax(input_view_inner_nodiag*t, dim = 1)
    # y_sm = F.softmax(target_view_inner_nodiag*t, dim=1)
    # y_logsm = F.log_softmax(target_view_inner_nodiag*t, dim = 1)
    # loss = (y_sm*(y_logsm - x_logsm)).sum(1).mean()
    loss = ((input_view_inner - target_view_inner)**2).sum(1).mean()
    # loss = F.l1_loss(input_view_inner, target_view_inner, reduction='sum')/ input_view_inner.shape[0]
    return loss

def contrastive_loss(feat1,feat2):
    # positive logits: Nx1
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1).detach()
    
    # l_pos = torch.einsum('nc,nc->n', [feat1, feat2]).unsqueeze(-1)
    # negative logits: NxK
    logits = torch.einsum('nc,ck->nk', [feat1, feat2.t()])
    batch = logits.shape[0]

    eps = torch.eye(batch, device=logits.device)
    eps[eps==1] = -0.1
    eps[eps==0] = 0.1

    labels = torch.arange(batch, device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits+eps, labels, reduction="mean")
    return loss
    

def shape_cpmt_cross_modal_ce(x1,y1,outputs, w=None):
    
    with torch.no_grad():
        batch_size = y1.shape[0]
        # rgb_shape_pow = torch.pow(outputs['shape']['zp'][:x1.shape[0]], 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        # ir_shape_pow = torch.pow(outputs['shape']['zp'][x1.shape[0]:], 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size).t()
        # rgb_ir_shape_cossim = rgb_shape_pow + ir_shape_pow
        # rgb_ir_shape_cossim.addmm_(outputs['shape']['zp'][:x1.shape[0]], outputs['shape']['zp'][x1.shape[0]:].t(), beta=1, alpha=-2)
        
        rgb_shape_normed = F.normalize(outputs['shape']['zp'][:x1.shape[0]], p=2, dim=1)
        ir_shape_normed = F.normalize(outputs['shape']['zp'][x1.shape[0]:], p=2, dim=1)
        rgb_ir_shape_cossim = torch.mm(rgb_shape_normed,ir_shape_normed.t())
        mask = y1.expand(batch_size,batch_size).eq(y1.expand(batch_size, batch_size).t())
        target4rgb, target4ir = [], []
        # target4rgb2, target4ir2 = [], []
        # target4rgb3, target4ir3 = [], []
        # target4rgb4, target4ir4 = [], []
        idx_temp = torch.arange(batch_size)
        for i in range(batch_size):
            sorted_idx_rgb = rgb_ir_shape_cossim[i][mask[i]].sort(descending=False)[1]
            sorted_idx_ir = rgb_ir_shape_cossim.t()[i][mask.t()[i]].sort(descending=False)[1]
            target4rgb.append(idx_temp[mask[i]][sorted_idx_rgb[0]].unsqueeze(0))
            # target4rgb2.append(idx_temp[mask[i]][sorted_idx_rgb[1]].unsqueeze(0))
            # target4rgb3.append(idx_temp[mask[i]][sorted_idx_rgb[2]].unsqueeze(0))
            # target4rgb4.append(idx_temp[mask[i]][sorted_idx_rgb[3]].unsqueeze(0))
            target4ir.append(idx_temp[mask.t()[i]][sorted_idx_ir[0]].unsqueeze(0))
            # target4ir2.append(idx_temp[mask.t()[i]][sorted_idx_ir[1]].unsqueeze(0))
            # target4ir3.append(idx_temp[mask.t()[i]][sorted_idx_ir[2]].unsqueeze(0))
            # target4ir4.append(idx_temp[mask.t()[i]][sorted_idx_ir[3]].unsqueeze(0))

        target4rgb = torch.cat(target4rgb)
        # target4rgb2 = torch.cat(target4rgb2)
        # target4rgb3 = torch.cat(target4rgb3)
        # target4rgb4 = torch.cat(target4rgb4)
        target4ir = torch.cat(target4ir)
        # target4ir2 = torch.cat(target4ir2)
        # target4ir3 = torch.cat(target4ir3)
        # target4ir4 = torch.cat(target4ir4)
    if w is None:
        loss_top1 = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:][target4rgb]) + sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]][target4ir])
    else:
        loss_top1 = reweight_sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:][target4rgb],w[:x1.shape[0]]) + reweight_sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]][target4ir], w[x1.shape[0]:])
    
    # loss_top2 = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:][target4rgb2]) + sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]][target4ir2])
    # loss_top3 = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:][target4rgb3]) + sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]][target4ir3])
    # loss_top4 = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:][target4rgb4]) + sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]][target4ir4])
    if w is None:
        # center1 = []
        # center2 = []
        # for i in range(0,y1.shape[0], 4):
        #     center1.append(outputs['rgbir']['logit2'][:y1.shape[0]][y1 == y1[i]].mean(0))
        #     center2.append(outputs['rgbir']['logit2'][y1.shape[0]:][y1 == y1[i]].mean(0))
        # center1 = torch.stack(center1)
        # center2 = torch.stack(center2)

        # loss_center2 = sce(center1, center2) + sce(center2, center1)
        # loss_kl_rgbir2 = loss_top1 + loss_center2
        loss_random = sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:])+sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]])
        # w1 = (loss_random.item())/(loss_random.item()+loss_top1.item())
        # w2 = (loss_top1.item())/(loss_random.item()+loss_top1.item())
        loss_kl_rgbir2 = loss_random+loss_top1
    # else:
    #     loss_kl_rgbir2 = loss_top1 + reweight_sce(outputs['rgbir']['logit2'][:x1.shape[0]],outputs['rgbir']['logit2'][x1.shape[0]:], w[:x1.shape[0]])+reweight_sce(outputs['rgbir']['logit2'][x1.shape[0]:],outputs['rgbir']['logit2'][:x1.shape[0]],w[x1.shape[0]:])

    return loss_kl_rgbir2



def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx