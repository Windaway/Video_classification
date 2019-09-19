import torch
import torchvision.transforms as transforms
from torch.utils import data
import os
from PIL import Image
import numpy as np
import config
from numba import jit
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from imgaug import augmenters as iaa
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


seq = iaa.Sequential([iaa.Resize(config.resize_wh),
    iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.2)),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.1)),
        iaa.CoarseDropout(
            (0.03, 0.15), size_percent=(0.02, 0.05),
            per_channel=0.2)
    ),

    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-25, 25),
        shear=0
    )],
        random_order=True)],
        random_order=False)

seq_val=iaa.Sequential([iaa.Resize(config.resize_wh)])


@jit(nopython=True)
def get_list(steps,path,files_list,mode=True):
    f_list=[]
    if mode:
        for x in range(config.step):
            t=x*steps+np.random.randint(0,steps)
            f_list.append(path+files_list[t])
    else:
        for x in range(config.step):
            t=x*steps+steps//2
            f_list.append(path+files_list[t])
    return f_list

def load_npy(path):
    t=np.load(path)
    return t

def get_array(file_list):
    l=len(file_list)
    t_array=np.zeros([1,l,config.image_wh,config.image_wh],np.uint8)
    for x in range(l):
        t=Image.open(file_list[x])
        t=np.array(t,np.uint8).reshape([config.image_wh,config.image_wh])
        t_array[0,x]=t
    t_array=np.transpose(t_array,[0,2,3,1])
    return t_array

def arr_trans(nparr):
    t=(config.step-1)*2
    temp_arr=np.zeros([config.resize_wh,config.resize_wh,t],np.float32)
    for x in range(config.step-1):
        temp_arr[:,:,2*x:2*x+2]=nparr[:,:,x:x+2]
    return temp_arr

class video_dataset(data.Dataset):
    def __init__(self,root='./',load_from_npy=False,train_npy='./train.npy',val_npy='./val.npy',save_npy=True,part=0.7,mode='Train',stack=True):
        super(video_dataset, self).__init__()
        self.root=root
        self.mode=mode
        self.stack=stack
        if load_from_npy:
            train_path=load_npy(train_npy)
            val_path=load_npy(val_npy)
        else:
            success_list=[]
            fail_list=[]
            all_dir=os.listdir(root)
            for x in all_dir:
                if x[:6]=='v_fail':
                    fail_list.append(x)
                if x[:6]=='v_succ':
                    success_list.append(x)
            l_success=len(success_list)
            l_fail=len(fail_list)
            success_list=np.array(success_list)
            fail_list=np.array(fail_list)
            idx_suc=np.arange(l_success)
            idx_suc=np.random.permutation(idx_suc)
            idx_fai=np.arange(l_fail)
            idx_fai=np.random.permutation(idx_fai)
            train_p_s=int(l_success*part)
            train_p_f=int(l_fail*part)
            train_path=np.hstack([success_list[idx_suc[:train_p_s]],fail_list[idx_fai[:train_p_f]]])
            val_path=np.hstack([success_list[idx_suc[train_p_s:]],fail_list[idx_fai[train_p_f:]]])
            if save_npy:
                np.save('./train.npy',train_path)
                np.save('./val.npy',val_path)
        if mode=='Train':
            self.file_list=train_path.tolist()
        else:
            self.file_list = val_path.tolist()
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        datapath = self.file_list[index]
        if datapath[:6]=='v_fail':
            label=0
        else:
            label=1
        # full_path=os.path.join(self.root,datapath)
        full_path = self.root+ datapath+'/'
        files=list(os.listdir(full_path))
        l_files=len(files)
        filenumber=l_files if l_files<config.max_files else config.max_files
        steps=filenumber//config.step
        files=get_list(steps,full_path,files,self.mode=='Train')
        file_array=get_array(files)
        if self.mode=='Train':
            file_array=seq(images=file_array)[0]
            file_array=np.array(file_array,np.float32)/255.
        else:
            file_array=seq_val(images=file_array)[0]
            file_array = np.array(file_array, np.float32) / 255.
        if self.stack:
            file_array=arr_trans(file_array)
        file_array=transforms.ToTensor()(file_array)
        return file_array,label





if __name__=='__main__':
    a=video_dataset(root='F:/DATA/',mode='Val')
    print(config.max_files)
    trainloader = data.DataLoader(a, batch_size=1,num_workers=1,shuffle=True)
    for i,data in enumerate(trainloader):
        img,label=data
        import cv2
        print(label.shape)
        print(label)
        a=img[0,7].numpy()
        print(a.shape)
        cv2.imshow('zzz',a)
        cv2.waitKey(0)



