import argparse
import sys
import time
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_bn import embed_net
from utils import *
from eval_sysu import eval_cross_cmc_map
import pdb

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality test sysumm01')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw, adp')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--date', default='')
parser.add_argument('--gpuversion', default= '3090', type=str)
path_dict = {}
path_dict['3090'] = ['/home/share/reid_dataset/SYSU-MM01/', '/home/share/fengjw/SYSU_MM01_SHAPE/']
path_dict['4090'] = ['/home/jiawei/data/SYSU-MM01/', '/home/jiawei/data/SYSU_MM01_SHAPE/']
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)


test_mode = [1, 2]  # thermal to visible
pool_dim = 2048
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

            feat_pool, feat_fc = get_feature(input, test_mode[0])
            # feat_pool, feat_fc = net(input, input, mode=test_mode[0])
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
            feat_pool, feat_fc = get_feature(input, test_mode[1])
            # feat_pool, feat_fc = net(input, input, mode=test_mode[1])
            query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1,device=img.device).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_feature(input, mode):
    feat_pool, feat_fc = net(input, input, mode=mode)
    input2 = fliplr(input)
    feat_pool2, feat_fc2 = net(input2, input2, mode=mode)
    feat_pool = (feat_pool+feat_pool2)/2
    feat_fc = (feat_fc+feat_fc2)/2

    return feat_pool, feat_fc
    

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_ema = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

end = time.time()
# training set
trainset = SYSUData(data_dir=path_dict[args.gpuversion][0],data_dir1=path_dict[args.gpuversion][1])
# testing set
query_img, query_label, query_cam = process_query_sysu(mode=args.mode,data_path_ori=path_dict[args.gpuversion][0])
gall_img, gall_label, gall_cam = process_gallery_sysu(mode=args.mode,data_path_ori=path_dict[args.gpuversion][0])

gallset  = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))

net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch)

net.to(device)
# cudnn.benchmark = True

# best two
net_path = 'save_model/sysu_step2085_p0.2intercutmix_bothcegkl_gradclip11.0_seed3_KL_1.0_p4_n8_lr_0.1_seed_3_ema_best.t'

net.load_state_dict(torch.load(net_path)['net'])
net.eval()

def test(net):
    # switch to evaluation mode



    net.eval()
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    for trial in range(10):

        gall_img, gall_label, gall_cam = process_gallery_sysu(mode=args.mode, trial=trial,data_path_ori=path_dict[args.gpuversion][0])

        trial_gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    print(
        'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            all_cmc[0], all_cmc[4], all_cmc[9], all_cmc[19], all_mAP, all_mINP))
    print(
        'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            all_cmc_pool[0], all_cmc_pool[4], all_cmc_pool[9], all_cmc_pool[19], all_mAP_pool, all_mINP_pool))



test(net)
