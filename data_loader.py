import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math
import os



class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
    
class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img
        
        

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        data_dir = '/home/share/reid_dataset/SYSU-MM01/'
        data_dir1 = '/home/share/fengjw/SYSU_MM01_SHAPE/'
        data_dir2 = '/home/share/fengjw/SYSU_MM01_MASK/'
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        train_color_image_shape = np.load(data_dir1 + 'train_rgb_resized_img.npy')
        self.train_color_label_shape = np.load(data_dir1 + 'train_rgb_resized_label.npy')

        train_thermal_image_shape = np.load(data_dir1 + 'train_ir_resized_img.npy')
        self.train_thermal_label_shape = np.load(data_dir1 + 'train_ir_resized_label.npy')
        print(train_color_image.shape, train_color_image_shape.shape)
        print(train_thermal_image.shape, train_thermal_image_shape.shape)

        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_color_image_shape   = train_color_image_shape
        self.train_thermal_image = train_thermal_image
        self.train_thermal_image_shape = train_thermal_image_shape
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
        self.transform_thermal_simple = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        self.transform_color_simple = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])           
        
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])
        
       
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1_shape,  target1_shape = self.train_color_image_shape[self.cIndex[index]],  self.train_color_label_shape[self.cIndex[index]]
        assert target1 == target1_shape
        img2_shape,  target2_shape = self.train_thermal_image_shape[self.tIndex[index]], self.train_thermal_label_shape[self.tIndex[index]]
        assert target2 == target2_shape
        
        if random.uniform(0, 1) > 0.5:
            trans_rgb = self.transform_color
        else:
            trans_rgb = self.transform_color1
        # trans_rgb = self.transform_color_simple

        img1 = trans_rgb(img1)
        img2 = self.transform_thermal(img2)

        img1_shape = self.transform_color_simple(img1_shape)
        img2_shape = self.transform_thermal_simple(img2_shape)

        return img1, img1_shape, img2, img2_shape, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = '/home/share/reid_dataset/RGB-IR_RegDB/'
        data_dir1 = '/home/share/fengjw/RegDB_shape/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        train_color_image_shape = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)

            
            img1 = Image.open(data_dir1+ color_img_file[i])
            img1 = img1.resize((144, 288), Image.ANTIALIAS)
            pix_array1 = np.array(img1)
            train_color_image_shape.append(pix_array1)
        train_color_image_shape = np.array(train_color_image_shape) 
        
        train_thermal_image = []
        train_thermal_image_shape = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)

            img1 = Image.open(data_dir1+ thermal_img_file[i])
            img1 = img1.resize((144, 288), Image.ANTIALIAS)
            pix_array1 = np.array(img1)
            train_thermal_image_shape.append(pix_array1)

        train_thermal_image_shape = np.array(train_thermal_image_shape)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label

        self.train_color_image_shape = train_color_image_shape  

        # BGR to RGB
        self.train_thermal_image_shape = train_thermal_image_shape
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_thermal_simple = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
        self.transform_color_simple = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]) 
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_shape = self.train_color_image_shape[self.cIndex[index]]
        img2_shape = self.train_thermal_image_shape[self.tIndex[index]]


        if random.uniform(0, 1) > 0.5:
            trans_rgb = self.transform_color
        else:
            trans_rgb = self.transform_color1
   
        img1 = trans_rgb(img1)
        img2 = self.transform_thermal(img2)

        img1_shape = self.transform_color_simple(img1_shape)
        img2_shape = self.transform_thermal_simple(img2_shape)
        
        return img1, img1_shape, img2, img2_shape, target1, target2

    def __len__(self):
        return len(self.train_color_label)



def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    T_pos = fname.find('T')
    D_pos = fname.find('D')
    F_pos = fname.find('F')
    camera = fname[D_pos:T_pos]
    picture = fname[F_pos+1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path

      
class VCM(object):
    def __init__(self, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        min_seq_len = 12
        data_dir = '/home/share/reid_dataset/HITSZ-VCM-UNZIP/'

        data_dir1 = '/home/share/fengjw/HITSZ-VCM-UNZIP_shape/'

        train_name_path = os.path.join(data_dir,'info/train_name.txt')
        track_train_info_path = os.path.join(data_dir,'info/track_train_info.txt')

        test_name_path = os.path.join(data_dir,'info/test_name.txt')
        track_test_info_path = os.path.join(data_dir,'info/track_test_info.txt')
        query_IDX_path = os.path.join(data_dir,'info/query_IDX.txt')

        # train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        # train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        # color_img_file, train_color_label = load_data(train_color_list)
        # thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_names = self._get_names(train_name_path)
        track_train = self._get_tracks(track_train_info_path)

        test_names = self._get_names(test_name_path)
        track_test = self._get_tracks(track_test_info_path)
        query_IDX =  self._get_query_idx(query_IDX_path)
        query_IDX -= 1

        track_query = track_test[query_IDX,:]
        print('query')
        print(track_query)
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        print('gallery')
        print(track_gallery)
        
        #---------visible to infrared-----------
        gallery_IDX_1 = self._get_query_idx(query_IDX_path)
        gallery_IDX_1 -= 1
        track_gallery_1 = track_test[gallery_IDX_1,:]

        query_IDX_1 = [j for j in range(track_test.shape[0]) if j not in gallery_IDX_1]
        track_query_1 = track_test[query_IDX_1,:]
        #-----------------------------------------

        train_ir, train_ir_shape, num_train_tracklets_ir,num_train_imgs_ir,train_rgb, train_rgb_shape, num_train_tracklets_rgb,num_train_imgs_rgb,num_train_pids,ir_label,rgb_label = \
          self._process_data_train(train_names,track_train,relabel=True,min_seq_len=min_seq_len, rootpath=data_dir, rootpath1=data_dir1)


        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data_test(test_names, track_query, relabel=False, min_seq_len=min_seq_len, rootpath=data_dir)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data_test(test_names, track_gallery, relabel=False, min_seq_len=min_seq_len, rootpath=data_dir)


        #--------visible to infrared-----------
        query_1, num_query_tracklets_1, num_query_pids_1, num_query_imgs_1 = \
          self._process_data_test(test_names, track_query_1, relabel=False, min_seq_len=min_seq_len, rootpath=data_dir)

        gallery_1, num_gallery_tracklets_1, num_gallery_pids_1, num_gallery_imgs_1 = \
          self._process_data_test(test_names, track_gallery_1, relabel=False, min_seq_len=min_seq_len, rootpath=data_dir)
        #---------------------------------------
        

        print("=> VCM loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_ir))
        print("train_rgb   | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_rgb))
        print("query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("---------------------------------")
        print("ir_label    | {}".format(np.unique(ir_label)))
        print("rgb_label   | {}".format(np.unique(rgb_label)))



        self.train_ir = train_ir
        self.train_ir_shape = train_ir_shape
        self.train_rgb = train_rgb
        self.train_rgb_shape = train_rgb_shape
        self.ir_label = ir_label
        self.rgb_label = rgb_label

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets
        self.num_gallery_tracklets = num_gallery_tracklets

        #------- visible to infrared------------
        self.query_1 = query_1
        self.gallery_1 = gallery_1

        self.num_query_pids_1 = num_query_pids_1
        self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_1
        self.num_gallery_tracklets_1 = num_gallery_tracklets_1
        #---------------------------------------


    def _get_names(self,fpath):
        """get image name, retuen name list"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _get_tracks(self,fpath):
        """get tracks file"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                names.append(tmp)
        names = np.array(names)
        return names


    def _get_query_idx(self, fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]


                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        print(idxs)
        return idxs  

    def _process_data_train(self,names,meta_data,relabel=False,min_seq_len=0, rootpath=None, rootpath1=None):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        # dict {pid : label}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        print('pid_list')
        print(pid_list)
        print(pid2label)
        tracklets_ir = []
        tracklets_ir_shape = []
        num_imgs_per_tracklet_ir = []
        ir_label = []

        tracklets_rgb = []
        tracklets_rgb_shape = []
        num_imgs_per_tracklet_rgb = []
        rgb_label = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            if m == 1:
                img_names = names[start_index-1:end_index]
                img_ir_paths = [os.path.join(rootpath,'Train',decoder_pic_path(img_name)) for img_name in img_names]
                img_ir_paths_shape = [os.path.join(rootpath1,'Train',decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_ir_paths) >= min_seq_len:
                    img_ir_paths = tuple(img_ir_paths)
                    ir_label.append(pid)
                    tracklets_ir.append((img_ir_paths,pid,camid))
                    # same id
                    num_imgs_per_tracklet_ir.append(len(img_ir_paths))
                    
                    # for shape
                    img_ir_paths_shape = tuple(img_ir_paths_shape)
                    tracklets_ir_shape.append((img_ir_paths_shape,pid,camid))
                    # same id
                    # num_imgs_per_tracklet_ir.append(len(img_ir_paths_shape))
            else:
                img_names = names[start_index-1:end_index]
                img_rgb_paths = [os.path.join(rootpath,'Train',decoder_pic_path(img_name)) for img_name in img_names]
                img_rgb_paths_shape = [os.path.join(rootpath1,'Train',decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_rgb_paths) >= min_seq_len:
                    img_rgb_paths = tuple(img_rgb_paths)
                    img_rgb_paths_shape = tuple(img_rgb_paths_shape)
                    rgb_label.append(pid)
                    tracklets_rgb.append((img_rgb_paths,pid,camid))
                    tracklets_rgb_shape.append((img_rgb_paths_shape,pid,camid))
                    #same id
                    num_imgs_per_tracklet_rgb.append(len(img_rgb_paths))

        num_tracklets_ir = len(tracklets_ir)
        num_tracklets_rgb = len(tracklets_rgb)
        num_tracklets = num_tracklets_rgb  + num_tracklets_ir

        return tracklets_ir, tracklets_ir_shape, num_tracklets_ir,num_imgs_per_tracklet_ir,tracklets_rgb, tracklets_rgb_shape, num_tracklets_rgb,num_imgs_per_tracklet_rgb,num_pids,ir_label,rgb_label

    def _process_data_test(self,names,meta_data,relabel=False,min_seq_len=0,rootpath=None):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        # dict {pid : label}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            img_names = names[start_index-1:end_index]
            img_paths = [os.path.join(rootpath,'Test',decoder_pic_path(img_name)) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet



import torch
class VideoDataset_test(data.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:  
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = range(num)
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
 
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
              
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list) 
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None: 
                    img = self.transform(img)

                imgs_ir.append(img.unsqueeze(0))
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_train(data.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir,dataset_rgb, dataset_ir_shape, dataset_rgb_shape, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_ir_shape = dataset_ir_shape
        self.dataset_rgb = dataset_rgb
        self.dataset_rgb_shape = dataset_rgb_shape
        self.seq_len = 3
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = self.transform #transforms.Compose( [
            # transforms.ToPILImage(),
            # transforms.Resize((288,144)),
            # transforms.Pad(10),
            # transforms.RandomCrop((288, 144)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
            # ChannelRandomErasing(probability = 0.5),
            # ChannelAdapGray(probability =0.5)])
            
        self.transform_thermal_simple = self.transform#transforms.Compose( [
            # transforms.ToPILImage(),
            # transforms.Resize((288,144)),
            # transforms.Pad(10),
            # transforms.RandomCrop((288, 144)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize
            # ])
        self.transform_color_simple = self.transform #transforms.Compose( [
            # transforms.ToPILImage(),
            # transforms.Resize((288,144)),
            # transforms.Pad(10),
            # transforms.RandomCrop((288, 144)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize
            # ]) 
            
        self.transform_color = self.transform#transforms.Compose( [
            # transforms.ToPILImage(),
            # transforms.Resize((288,144)),
            # transforms.Pad(10),
            # transforms.RandomCrop((288, 144)),
            # transforms.RandomHorizontalFlip(),
            # # transforms.RandomGrayscale(p = 0.1),
            # transforms.ToTensor(),
            # normalize,
            # ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = self.transform#transforms.Compose( [
            # transforms.ToPILImage(),
            # transforms.Resize((288,144)),
            # transforms.Pad(10),
            # transforms.RandomCrop((288, 144)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
            # ChannelRandomErasing(probability = 0.5),
            # ChannelExchange(gray = 2)])


    def __len__(self):
        return len(self.dataset_rgb)


    def __getitem__(self, index):

        if random.uniform(0, 1) > 0.5:
            trans_rgb = self.transform_color
        else:
            trans_rgb = self.transform_color1

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        img_ir_paths_shape, pid_ir_shape, camid_ir_shape = self.dataset_ir_shape[self.index2[index]]

        num_ir = len(img_ir_paths)

        img_rgb_paths,pid_rgb,camid_rgb = self.dataset_rgb[self.index1[index]]
        img_rgb_paths_shape, pid_rgb_shape, camid_rgb_shape = self.dataset_rgb_shape[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        idx1 = np.random.choice(num_ir, self.seq_len)
        imgs_ir = []
        imgs_ir_shape = []
        for index in idx1:
            index = int(index)
            img_path = img_ir_paths[index]
            img_path_shape = img_ir_paths_shape[index]
            img = read_image(img_path)
            img_shape = read_image(img_path_shape)
            img = np.array(img)
            img_shape = np.array(img_shape)
            img = self.transform_thermal(img)
            img_shape = self.transform_thermal_simple(img_shape)

            imgs_ir.append(img.unsqueeze(0))
            imgs_ir_shape.append(img_shape.unsqueeze(0))
        imgs_ir = torch.cat(imgs_ir, dim=0)        
        imgs_ir_shape = torch.cat(imgs_ir_shape, dim=0)        

        idx2 = np.random.choice(num_rgb, self.seq_len)
        imgs_rgb = []
        imgs_rgb_shape = []
        for index in idx2:
            index = int(index)
            img_path = img_rgb_paths[index]
            img_path_shape = img_rgb_paths_shape[index]
            img = read_image(img_path)
            img_shape = read_image(img_path_shape)
            img = np.array(img)
            img_shape = np.array(img_shape)

            img = trans_rgb(img)
            img_shape = self.transform_color_simple(img_shape)

            imgs_rgb.append(img.unsqueeze(0))
            imgs_rgb_shape.append(img_shape.unsqueeze(0))
        imgs_rgb = torch.cat(imgs_rgb, dim=0)        
        imgs_rgb_shape = torch.cat(imgs_rgb_shape, dim=0)   
        pid_ir = torch.tensor(pid_ir).repeat(self.seq_len)     
        pid_rgb = torch.tensor(pid_rgb).repeat(self.seq_len)



        return imgs_rgb, imgs_rgb_shape, imgs_ir, imgs_ir_shape, pid_rgb, pid_ir





class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataSYSU(data.Dataset):
    def __init__(self, test_img_file, test_label, test_img_file_shape, test_label_shape, transform=None, img_size = (144,288)):

        test_image = []
        test_image_shape = []
        assert len(test_img_file) == len(test_img_file_shape)
        for i in range(len(test_img_file)):
            assert test_label[i] == test_label_shape[i]
            img = Image.open(test_img_file[i])
            img_shape = Image.open(test_img_file_shape[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            img_shape = img_shape.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            pix_array_shape = np.array(img_shape)
            test_image.append(pix_array)
            test_image_shape.append(pix_array_shape)
        test_image = np.array(test_image)
        test_image_shape = np.array(test_image_shape)
        self.test_image = test_image
        self.test_image_shape = test_image_shape
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img2 = self.test_image_shape[index]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target1

    def __len__(self):
        return len(self.test_image)
   
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    return file_image, file_label




if __name__ == '__main__':
    dataset = VCM()