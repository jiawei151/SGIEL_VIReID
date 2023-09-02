import numpy as np
from PIL import Image
import pdb
import os


data_path = '/home/share/reid_dataset/SYSU-MM01/'
data_path1 = '/home/share/fengjw/SYSU_MM01_SHAPE/'
data_path2 = '/home/share/fengjw/SYSU-MM01_SHAPE_composepow/'


rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load id info
file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
    print(len(ids))
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    print(len(ids))
# combine train and val split   
id_train.extend(id_val) 

files_rgb = []
files_rgb_shape = []
files_rgb_mask = []
files_ir = []
files_ir_shape = []
files_ir_mask = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
        img_dir1 = os.path.join(data_path1,cam,id)
        if os.path.isdir(img_dir1):
            new_files = sorted([img_dir1+'/'+i for i in os.listdir(img_dir1)])
            files_rgb_shape.extend(new_files)
        img_dir2 = os.path.join(data_path2,cam,id)
        if os.path.isdir(img_dir2):
            new_files = sorted([img_dir2+'/'+i for i in os.listdir(img_dir2)])
            files_rgb_mask.extend(new_files)
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
        img_dir1 = os.path.join(data_path1,cam,id)
        if os.path.isdir(img_dir1):
            new_files = sorted([img_dir1+'/'+i for i in os.listdir(img_dir1)])
            files_ir_shape.extend(new_files)
        img_dir2 = os.path.join(data_path2,cam,id)
        if os.path.isdir(img_dir2):
            new_files = sorted([img_dir2+'/'+i for i in os.listdir(img_dir2)])
            files_ir_mask.extend(new_files)
for i in range(len(files_rgb)):
    if not files_rgb[i][-19:-1] == files_rgb_mask[i][-19:-1]:
        import pdb
        pdb.set_trace()
    if not files_rgb[i][-19:-1] == files_rgb_shape[i][-19:-1]:
        import pdb
        pdb.set_trace()
for i in range(len(files_ir)):
    if not files_ir[i][-19:-1] == files_ir_mask[i][-19:-1]:
        import pdb
        pdb.set_trace()
    if not files_ir[i][-19:-1] == files_ir_shape[i][-19:-1]:
        import pdb
        pdb.set_trace()
# relabel
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
print(len(pid_container))
pid2label = {pid:label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288
def read_imgs(train_image):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array) 
        
        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)

# rgb imges
rgb_img, rgb_label = read_imgs(files_rgb)
rgb_img_shape, rgb_label_shape = read_imgs(files_rgb_shape)
train_img, train_label = read_imgs(files_ir)
train_img_shape, train_label_shape = read_imgs(files_ir_shape)

def func(img, img_shape):
    res = []
    shape_mean = img_shape.mean()
    imgsize = 288*144
    for i in range(img.shape[0]):
        tmp = (img_shape[i][:,:,0]>shape_mean).sum()/imgsize
        res.append(tmp)
    return np.array(res)

res_rgb = func(rgb_img, rgb_img_shape)
res_ir = func(train_img, train_img_shape)
thres_rgb = 0.1
thres_ir = 0.01

np.save(data_path + 'train_rgb_resized_img_new.npy', rgb_img[res_rgb>thres_rgb])
np.save(data_path + 'train_rgb_resized_label_new.npy', rgb_label[res_rgb>thres_rgb])

np.save(data_path1 + 'train_rgb_resized_img_new.npy', rgb_img_shape[res_rgb>thres_rgb])
np.save(data_path1 + 'train_rgb_resized_label_new.npy', rgb_label_shape[res_rgb>thres_rgb])

# ir imges


np.save(data_path + 'train_ir_resized_img_new.npy', train_img[res_ir>thres_ir])
np.save(data_path + 'train_ir_resized_label_new.npy', train_label[res_ir>thres_ir])

np.save(data_path1 + 'train_ir_resized_img_new.npy', train_img_shape[res_ir>thres_ir])
np.save(data_path1 + 'train_ir_resized_label_new.npy', train_label_shape[res_ir>thres_ir])

