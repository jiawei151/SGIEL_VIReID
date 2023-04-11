from __future__ import print_function
import argparse
import sys
import time
from sklearn.cluster import estimate_bandwidth
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestDataSYSU, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_bn import embed_net
from utils import *
import copy
from loss import OriTripletLoss, TripletLoss_WRT, KLDivLoss, TripletLoss_ADP, pdist_torch, reweight_sce, sce, exchange_sce, shape_cpmt_cross_modal_ce
# from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import pdb

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--margin_shape', default=0.1, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--augc', default=0 , type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default= 0 , type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default= 1. , type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1 , type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1 , type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default= 1 , type=int,
                    metavar='square', help='gamma for the hard mining')
parser.add_argument('--date', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--delta', default= 1e-3 , type=float,
                    metavar='delta', help='gamma for the hard mining')
      
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path_shape = '/home/share/fengjw/SYSU_MM01_SHAPE/'
    data_path = '/home/share/reid_dataset/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/home/share/reid_dataset/RGB-IR_RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [1, 2]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'adp':
    suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
else:
    suffix = suffix + '_{}'.format(args.method)
suffix = suffix + '_KL_{}'.format(args.kl)
# suffix = suffix + '_smargin_{}'.format(args.margin_shape)
if args.augc==1:
    suffix = suffix + '_aug_G'  
if args.rande>0:
    suffix = suffix + '_erase_{}'.format( args.rande)
    
suffix = suffix + '_p{}_n{}_lr_{}_seed_{}'.format( args.num_pos, args.batch_size, args.lr, args.seed)  

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + args.date + '/' + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + args.date + '/' + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
# writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_ema = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]
    
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande>0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability = args.rande)]

if args.augc ==1:
    # transform_train_list = transform_train_list +  [ChannelAdap(probability =0.5)]
    transform_train_list = transform_train_list + [ChannelAdapGray(probability =0.5)]
    
transform_train = transforms.Compose( transform_train_list )

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam, query_img_shape, query_label_shape, query_cam_shape = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam, gall_img_shape, gall_label_shape, gall_cam_shape = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

gallset  = TestDataSYSU(gall_img, gall_label, gall_img_shape, gall_label_shape, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestDataSYSU(query_img, query_label, query_img_shape, query_label_shape, transform=transform_test, img_size=(args.img_w, args.img_h))
# gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
# queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..', args.method)

net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch)
net_ema = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch)
print('use model without nonlocal but gmpool')

net.to(device)
net_ema.to(device)
# cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
    print('trip: wrt')
    # loader_batch = args.batch_size * args.num_pos
    # criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
elif args.method == 'adp':
    criterion_tri = TripletLoss_ADP(alpha = args.alpha, gamma = args.gamma, square = args.square)
    print('trip: adp')

else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
    # criterion_tri_shape= OriTripletLoss(batch_size=loader_batch, margin=args.margin_shape)
    print('trip: ori')
# criterion_kl = KLDivLoss()
criterion_id.to(device)
criterion_tri.to(device)
# criterion_tri_shape.to(device)
# criterion_kl.to(device)
if args.optim == 'sgd':
    ignored_params = list(map(id, net.classifier.parameters()))
    ignored_params += list(map(id, net.bottleneck.parameters()))
    ignored_params += list(map(id, net.bottleneck_ir.parameters()))
    ignored_params += list(map(id, net.classifier_ir.parameters())) 
    # print('#####larger lr for ir#####')
    if hasattr(net,'classifier_shape'):
        ignored_params += list(map(id, net.classifier_shape.parameters())) 
        ignored_params += list(map(id, net.bottleneck_shape.parameters())) 
        
        ignored_params += list(map(id, net.projs.parameters())) 
        print('#####larger lr for shape#####')

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    params = [{'params': base_params, 'lr': 0.1 * args.lr}, {'params': net.classifier.parameters(), 'lr': args.lr},]
    
    params.append({'params': net.bottleneck.parameters(), 'lr': args.lr})
    params.append({'params': net.bottleneck_shape.parameters(), 'lr': args.lr})
    params.append({'params': net.classifier_ir.parameters(), 'lr': args.lr})
    params.append({'params': net.classifier_shape.parameters(), 'lr': args.lr})
    params.append({'params': net.projs.parameters(), 'lr': args.lr})
    params.append({'params': net.bottleneck_ir.parameters(), 'lr': args.lr})

    # optimizer = optim.Adam(params, weight_decay=5e-4)
    optimizer = optim.SGD(params, weight_decay=5e-4, momentum=0.9, nesterov=True)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
        ema_w = 1000
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
        ema_w = 1000
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
        ema_w = 1000 * args.lr / lr
    elif epoch >= 50 and epoch < 100:
        lr = args.lr * 0.01
        ema_w = 1000 * args.lr / lr
    elif epoch >= 100:
        lr = args.lr * 0.001
        ema_w = 1000 * args.lr / lr
    optimizer.param_groups[0]['lr'] = 0.1*lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr, ema_w


def update_ema_variables(net, net_ema, alpha, global_step=None):
        with torch.no_grad():
            for ema_item, new_item in zip(net_ema.named_parameters(), net.named_parameters()):
                ema_key, ema_param = ema_item
                new_key, new_param = new_item
                if 'classifier' in ema_key or 'bottleneck' in ema_key or 'projs' in ema_key:
                    alpha_now = alpha*2
                else:
                    alpha_now = alpha
                mygrad = new_param.data - ema_param.data
                ema_param.data.add_(mygrad, alpha=alpha_now)

def train(epoch):

    current_lr, ema_w = adjust_learning_rate(optimizer, epoch)
    print('current lr', current_lr)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    id_loss_shape = AverageMeter()
    id_loss_shape2 = AverageMeter()
    mutual_loss = AverageMeter()
    mutual_loss2 = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    net_ema.train()
    end = time.time()

    for batch_idx, (x1, x1_shape, x2, x2_shape, y1, y2) in enumerate(trainloader):

        y = torch.cat((y1, y2), 0)
        x1, x1_shape, x2, x2_shape, y1, y2, y = x1.cuda(), x1_shape.cuda(), x2.cuda(), x2_shape.cuda(), y1.cuda(), y2.cuda(), y.cuda()
                
        data_time.update(time.time() - end)


        outputs = net(x1, x2, x1_shape, x2_shape, y=y)
        with torch.no_grad():
            outputs_ema = net_ema(x1, x2, x1_shape, x2_shape, y=y)

        # id loss
        loss_id = criterion_id(outputs['rgbir']['logit'], y)
        loss_id2 = criterion_id(outputs['rgbir']['logit2'], y)
        loss_id_shape = criterion_id(outputs['shape']['logit'], y)
        
        # triplet loss
        loss_tri, batch_acc = criterion_tri(outputs['rgbir']['bef'], y)
        
        # cross modal distill
        loss_kl_rgbir = sce(outputs['rgbir']['logit'][:x1.shape[0]],outputs['rgbir']['logit'][x1.shape[0]:])+sce(outputs['rgbir']['logit'][x1.shape[0]:],outputs['rgbir']['logit'][:x1.shape[0]])

        # shape complementary
        loss_kl_rgbir2 = shape_cpmt_cross_modal_ce(x1, y1, outputs)

        # shape consistent
        loss_estimate =  ((outputs['rgbir']['zp']-outputs_ema['shape']['zp'].detach()) ** 2).mean(1).mean() + sce(torch.mm(outputs['rgbir']['zp'], net.classifier_shape.weight.data.detach().t()), outputs_ema['shape']['logit'])
        # loss_estimate =  ((outputs['rgbir']['zp']-outputs['shape']['zp'].detach()) ** 2).mean(1).mean() + sce(net.classifier_shape(outputs['rgbir']['zp']), outputs['shape']['logit'])
        

        ############## reweighting ###############
        compliment_grad = torch.autograd.grad(loss_id2+loss_kl_rgbir2, outputs['rgbir']['bef'], retain_graph=True)[0]
        consistent_grad = torch.autograd.grad(loss_estimate, outputs['rgbir']['bef'], retain_graph=True)[0]

        with torch.no_grad():
            compliment_grad_norm = (compliment_grad.norm(p=2,dim=-1)).mean()
            consistent_grad_norm = (consistent_grad.norm(p=2,dim=-1)).mean()
            w1 = consistent_grad_norm / (compliment_grad_norm+consistent_grad_norm) * 2
            w2 = compliment_grad_norm / (compliment_grad_norm+consistent_grad_norm) * 2  

        ############## orthogonalize loss ###############
        proj_inner = torch.mm(F.normalize(net.projs[0], 2, 0).t(), F.normalize(net.projs[0], 2, 0))
        eye_label = torch.eye(net.projs[0].shape[1],device=device)
        loss_ortho = (proj_inner - eye_label).abs().sum(1).mean()



        loss = loss_id + loss_tri + loss_id_shape + loss_kl_rgbir + w1*loss_estimate + w2*loss_id2 +w2*loss_kl_rgbir2 + loss_ortho

        if not check_loss(loss):
            import pdb
            pdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12.)
        optimizer.step()
        


        update_ema_variables(net, net_ema, 1/ema_w)


        # update P
        train_loss.update(loss_id2.item(), 2 * x1.size(0))
        id_loss.update(loss_id.item(), 2 * x1.size(0))
        id_loss_shape.update(loss_id_shape.item(), 2 * x1.size(0))
        id_loss_shape2.update(w1.item(), 2 * x1.size(0))
        mutual_loss2.update(loss_kl_rgbir2.item(), 2 * x1.size(0))
        mutual_loss.update(loss_ortho.item(), 2 * x1.size(0))
        # kl_loss.update(loss_kl2.item()+loss_kl.item() , 2 * x1.size(0))
        kl_loss.update(loss_estimate.item(), 2 * x1.size(0))
        total += y.size(0)

        # measure elapsed time   100. * correct / total
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            # import pdb
            # pdb.set_trace()
            print('Epoch:[{}][{}/{}]'
                  'L:{id_loss.val:.4f}({id_loss.avg:.4f}) '
                  'L2:{train_loss.val:.4f}({train_loss.avg:.4f}) '
                  'sL:{id_loss_shape.val:.4f}({id_loss_shape.avg:.4f}) '
                  'w1:{id_loss_shape2.val:.4f}({id_loss_shape2.avg:.4f}) '
                  'or:{mutual_loss.val:.4f}({mutual_loss.avg:.4f}) '
                  'ML2:{mutual_loss2.val:.4f}({mutual_loss2.avg:.4f}) '
                  'KL:{kl_loss.val:.4f}({kl_loss.avg:.4f}) '.format(
                epoch, batch_idx, len(trainloader),
                train_loss=train_loss, id_loss=id_loss, id_loss_shape=id_loss_shape, id_loss_shape2=id_loss_shape2, mutual_loss=mutual_loss, mutual_loss2=mutual_loss2, kl_loss=kl_loss))




def test(epoch, net):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, mode=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            # feat, feat_att = net(input, input, mode=test_mode[1])
            feat, feat_att = net(input, input, mode=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    if dataset == 'regdb':
        distmat2 = np.matmul(gall_feat, np.transpose(query_feat))
        distmat_att2 = np.matmul(gall_feat_att, np.transpose(query_feat_att))


    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc2, mAP2, mINP2      = eval_regdb(-distmat2, gall_label, query_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
        cmc_att2, mAP_att2, mINP_att2  = eval_regdb(-distmat_att2, gall_label, query_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmc2, mAP2, mINP2, cmc_att2, mAP_att2, mINP_att2

def test_shape(epoch, net):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_shape = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    gall_feat_att_shape = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, input_shape, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input_shape = Variable(input_shape.cuda())
            feat, feat_att = net(input, input, mode=test_mode[0])
            # feat_shape, feat_att_shape = net(input_shape, input_shape, input_shape, input_shape, mode=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            # gall_feat_shape[ptr:ptr + batch_num, :] = feat_shape.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            # gall_feat_att_shape[ptr:ptr + batch_num, :] = feat_att_shape.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_shape = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    query_feat_att_shape = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, input_shape, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            input_shape = Variable(input_shape.cuda())
            feat, feat_att = net(input, input, mode=test_mode[1])
            # feat_shape, feat_att_shape = net(input_shape, input_shape, input_shape, input_shape, mode=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            # query_feat_shape[ptr:ptr + batch_num, :] = feat_shape.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            # query_feat_att_shape[ptr:ptr + batch_num, :] = feat_att_shape.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    # distmat_shape = np.matmul(query_feat_shape, np.transpose(gall_feat_shape))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    # distmat_att_shape = np.matmul(query_feat_att_shape, np.transpose(gall_feat_att_shape))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        # cmcs, mAPs, mINPs = eval_sysu(-distmat_shape, query_label_shape, gall_label_shape, query_cam_shape, gall_cam_shape)
        # cmc_atts, mAP_atts, mINP_atts = eval_sysu(-distmat_att_shape, query_label_shape, gall_label_shape, query_cam_shape, gall_cam_shape)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))


    cmcs, mAPs, mINPs, cmc_atts, mAP_atts, mINP_atts = 0,0,0,0,0,0
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmcs, mAPs, mINPs, cmc_atts, mAP_atts, mINP_atts
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# training
print('==> Start Training...')
for epoch in range(start_epoch, 120 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    # print(trainset.cIndex)
    # print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    # if epoch == start_epoch:
    #     pretrain(epoch)
    if epoch == 0:
        net_ema.load_state_dict(net.state_dict())
        print('init ema modal')


    train(epoch)

    print('Test Epoch: {}'.format(epoch))

    # testing
    # cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
    cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmcs, mAPs, mINPs, cmc_atts, mAP_atts, mINP_atts = test_shape(epoch, net)
    # wandb.log({'rank1': cmc[0],
    #             'mAP': mAP,
    #             'rank1att': cmc_att[0],
    #             'mAPatt': mAP_att,
    #             },step=epoch)
    # wandb.log({'rank1_shape': cmcs[0],
    #             'mAPs_shape': mAPs,
    #             'rank1atts_shape': cmc_atts[0],
    #             'mAPatt_shape': mAP_atts,
    #             },step=epoch)
    cmc_ema, mAP_ema, mINP_ema, cmc_att_ema, mAP_att_ema, mINP_att_ema, cmcs_ema, mAPs_ema, mINPs_ema, cmc_atts_ema, mAP_atts_ema, mINP_atts_ema = test_shape(epoch, net_ema)
    # wandb.log({'rank1_ema': cmc_ema[0],
    #             'mAP_ema': mAP_ema,
    #             'rank1att_ema': cmc_att_ema[0],
    #             'mAPatt_ema': mAP_att_ema,
    #             },step=epoch)
    # wandb.log({'rank1_shape_ema': cmcs_ema[0],
    #             'mAPs_shape_ema': mAPs_ema,
    #             'rank1atts_shape_ema': cmc_atts_ema[0],
    #             'mAPatt_shape_ema': mAP_atts_ema,
    #             },step=epoch)
    # save model
    if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
        best_acc = cmc_att[0]
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'cmc': cmc_att,
            'mAP': mAP_att,
            'mINP': mINP_att,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_best.t')
    if cmc_att_ema[0] > best_acc_ema:  # not the real best for sysu-mm01
        best_acc_ema = cmc_att_ema[0]
        best_epoch_ema = epoch
        state = {
            'net': net_ema.state_dict(),
            'cmc': cmc_att_ema,
            'mAP': mAP_att_ema,
            'mINP': mINP_att_ema,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_ema_best.t')

    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
    print('Best Epoch [{}]'.format(best_epoch))
      
    print('------------------ema eval------------------')
    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_ema[0], cmc_ema[4], cmc_ema[9], cmc_ema[19], mAP_ema, mINP_ema))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att_ema[0], cmc_att_ema[4], cmc_att_ema[9], cmc_att_ema[19], mAP_att_ema, mINP_att_ema))
    print('Best Epoch [{}]'.format(best_epoch_ema))
