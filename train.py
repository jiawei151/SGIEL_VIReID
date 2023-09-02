from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import SYSUData, TestData
from data_manager import *
from eval_metrics import eval_sysu
from model_bn import embed_net
from utils import *
import copy
from loss import OriTripletLoss, TripletLoss_WRT, TripletLoss_ADP, sce, shape_cpmt_cross_modal_ce
# from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import pdb
import wandb

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
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=3, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--date', default='12.22', help='date of exp')

parser.add_argument('--gradclip', default= 11, type=float,
            metavar='gradclip', help='gradient clip')
parser.add_argument('--gpuversion', default= '3090', type=str, help='3090 or 4090')
path_dict = {}
path_dict['3090'] = ['/home/share/reid_dataset/SYSU-MM01/', '/home/share/fengjw/SYSU_MM01_SHAPE/']
path_dict['4090'] = ['/home/jiawei/data/SYSU-MM01/', '/home/jiawei/data/SYSU_MM01_SHAPE/']
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
wandb.init(config=args, project='rgbir-reid2')
args.method = args.method + "_gradclip" + str(args.gradclip) + "_seed" + str(args.seed)
wandb.run.name = args.method
# set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible


checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
# if args.method == 'adp':
#     suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
# else:
suffix = suffix + '_{}'.format(args.method)
  
suffix = suffix + '_p{}_n{}_lr_{}'.format( args.num_pos, args.batch_size, args.lr)  

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim


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


transform_train = transforms.Compose( transform_train_list )

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_dir=path_dict[args.gpuversion][0],data_dir1=path_dict[args.gpuversion][1])
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(mode=args.mode,data_path_ori=path_dict[args.gpuversion][0])
    # gall_img, gall_label, gall_cam = process_gallery_sysu_all(mode=args.mode,data_path_ori=path_dict[args.gpuversion][0])
    gall_img, gall_label, gall_cam = process_gallery_sysu(mode=args.mode, trial=0, data_path_ori=path_dict[args.gpuversion][0]) 

set_seed(args.seed)



gallset  = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h))

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
if 'agw' in args.method:
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_id.to(device)
criterion_tri.to(device)
if args.optim == 'sgd':
    ignored_params = list(map(id, net.classifier.parameters()))
    ignored_params += list(map(id, net.bottleneck.parameters()))
    ignored_params += list(map(id, net.bottleneck_ir.parameters()))
    ignored_params += list(map(id, net.classifier_ir.parameters())) 
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    ema_w = 1000
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 85:
        lr = args.lr * 0.1
        ema_w = 10000
    elif epoch < 120:
        lr = args.lr * 0.01
        ema_w = 100000
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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

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
    for batch_idx, (inputs) in enumerate(trainloader):
        x1, x1_shape, x2, x2_shape, y1, y2 = inputs
        y = torch.cat((y1, y2), 0)
        x1, x1_shape, x2, x2_shape, y1, y2, y = x1.cuda(), x1_shape.cuda(), x2.cuda(), x2_shape.cuda(), y1.cuda(), y2.cuda(), y.cuda()
                
        data_time.update(time.time() - end)

        cutmix_prob = np.random.rand(1)
        if cutmix_prob < 0.2:
        # generate mixed sample
            x = torch.cat((x1, x2), 0)
            x_shape = torch.cat((x1_shape, x2_shape), 0)
            lam = np.random.beta(1, 1)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

            rand_index = torch.randperm(y1.size()[0]).cuda()
            target_a = y
            target_b = torch.cat((y2[rand_index],y1[rand_index]), 0)
            x[:, :, bbx1:bbx2, bby1:bby2] = torch.cat((x2[rand_index, :, bbx1:bbx2, bby1:bby2],x1[rand_index, :, bbx1:bbx2, bby1:bby2]),0)
            x_shape[:, :, bbx1:bbx2, bby1:bby2] = torch.cat((x2_shape[rand_index, :, bbx1:bbx2, bby1:bby2],x1_shape[rand_index, :, bbx1:bbx2, bby1:bby2]),0)

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            outputs = net(x[:y1.shape[0]], x[y1.shape[0]:], x_shape[:y1.shape[0]], x_shape[y1.shape[0]:])
            with torch.no_grad():
                outputs_ema = net_ema(x[:y1.shape[0]], x[y1.shape[0]:], x_shape[:y1.shape[0]], x_shape[y1.shape[0]:])
            
            loss_id = criterion_id(outputs['rgbir']['logit'], target_a) * lam + criterion_id(outputs['rgbir']['logit'], target_b) * (1. - lam)
            loss_id2 = torch.tensor([0]).cuda()
            loss_id_shape = criterion_id(outputs['shape']['logit'], target_a) * lam + criterion_id(outputs['shape']['logit'], target_b) * (1. - lam)
            loss_tri = torch.tensor([0]).cuda() 
            loss_kl_rgbir = sce(outputs['rgbir']['logit'][:x1.shape[0]],outputs['rgbir']['logit'][x1.shape[0]:])+sce(outputs['rgbir']['logit'][x1.shape[0]:],outputs['rgbir']['logit'][:x1.shape[0]])
            w1 = torch.tensor([1.]).cuda()
            loss_estimate = torch.tensor([0]).cuda()
            w2 = torch.tensor([1.]).cuda()
            loss_kl_rgbir2 = torch.tensor([0]).cuda()

        else:
            with torch.no_grad():
                outputs_ema = net_ema(x1, x2, x1_shape, x2_shape)
            outputs = net(x1, x2, x1_shape, x2_shape)

            # id loss
            # if epoch < 40:
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradclip)
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


def test(net):
    pool_dim = 2048
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1,device=img.device).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip
    def extract_gall_feat(gall_loader):
        net.eval()
        print ('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        ngall = len(gall_loader.dataset)
        gall_feat_pool = np.zeros((ngall, pool_dim))
        gall_feat_fc = np.zeros((ngall, pool_dim))
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, mode=test_mode[0])
                input2 = fliplr(input)
                feat_pool2, feat_fc2 = net(input2, input2, mode=test_mode[0])
                feat_pool = (feat_pool+feat_pool2)/2
                feat_fc = (feat_fc+feat_fc2)/2

                gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
                gall_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time()-start))
        return gall_feat_pool, gall_feat_fc
    
    def extract_query_feat(query_loader):
        net.eval()
        print ('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        nquery = len(query_loader.dataset)
        query_feat_pool = np.zeros((nquery, pool_dim))
        query_feat_fc = np.zeros((nquery, pool_dim))
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, mode=test_mode[1])
                input2 = fliplr(input)
                feat_pool2, feat_fc2 = net(input2, input2, mode=test_mode[1])
                feat_pool = (feat_pool+feat_pool2)/2
                feat_fc = (feat_fc+feat_fc2)/2
                query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
                query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num         
        print('Extracting Time:\t {:.3f}'.format(time.time()-start))
        return query_feat_pool, query_feat_fc
    # switch to evaluation mode
    net.eval()
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)

    # gall_img, gall_label, gall_cam = process_gallery_sysu(mode=args.mode, trial=0)

    trial_gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

    # pool5 feature
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

    # fc feature
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    all_cmc = cmc
    all_mAP = mAP
    all_mINP = mINP
    all_cmc_pool = cmc_pool
    all_mAP_pool = mAP_pool
    all_mINP_pool = mINP_pool
    return all_cmc, all_mAP 



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

    if epoch == 0:
        net_ema.load_state_dict(net.state_dict())
        print('init ema modal')


    train(epoch)

    print('Test Epoch: {}'.format(epoch))

    # testing
    cmc, mAP = test(net)
    wandb.log({'rank1': cmc[0],
                'mAP': mAP,
                },step=epoch)
    cmc_ema, mAP_ema = test(net_ema)
    wandb.log({'rank1_ema': cmc_ema[0],
                'mAP_ema': mAP_ema,
                },step=epoch)
    # save model
    if cmc[0] > best_acc: 
        best_acc = cmc[0]
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'cmc': cmc,
            'mAP': mAP,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_best.t')
    if cmc_ema[0] > best_acc_ema:  
        best_acc_ema = cmc_ema[0]
        best_epoch_ema = epoch
        state = {
            'net': net_ema.state_dict(),
            'cmc': cmc_ema,
            'mAP': mAP_ema,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_ema_best.t')
    if epoch % 5 == 0:
        state = {
            'net': net_ema.state_dict(),
        }
        torch.save(state, checkpoint_path + suffix + '_' + str(epoch) + '_.t')

    print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print('Best Epoch [{}]'.format(best_epoch))
      
    print('------------------ema eval------------------')
    print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc_ema[0], cmc_ema[4], cmc_ema[9], cmc_ema[19], mAP_ema))
    print('Best Epoch [{}]'.format(best_epoch_ema))
