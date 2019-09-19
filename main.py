import torch
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from absl import app
from absl import flags
import dataset
from torch.utils import data
from model import *
from utils import *
from matplotlib import pyplot as plt
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import pylab as pl
flags.DEFINE_float('lr', 1e-3,
                    'learn rate')
flags.DEFINE_string('save_dir', './ckpt/',
                    'save_dir')
flags.DEFINE_integer('epochs', 1000,
                    'epochs')
flags.DEFINE_integer('warm_epochs', 100,
                    'warm_epochs')
flags.DEFINE_string('model', 'RNN',
                    'model')
flags.DEFINE_string('ckpt_path', './',
                    'ckpt_path')
flags.DEFINE_boolean('restore', False,
                    'restore')
flags.DEFINE_integer('batch_size', 6,
                    'batch_size')
flags.DEFINE_boolean('first_train', True,
                    'first_train')


FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.first_train:
        pre_dataset=dataset.video_dataset('F:/DATA/',False)
    if not os.path.isdir(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)
    if FLAGS.model=='RNN':
        model=RNN()
        train_dataset = dataset.video_dataset('F:/DATA/', True, mode='Train',stack=True)
        val_dataset = dataset.video_dataset('F:/DATA/', True, mode='Val',stack=True)
    if FLAGS.model=='C2D':
        model=C2D()
        train_dataset = dataset.video_dataset('F:/DATA/', True, mode='Train',stack=False)
        val_dataset = dataset.video_dataset('F:/DATA/', True, mode='Val',stack=False)
    if FLAGS.model=='C3D':
        model=C3D()
        train_dataset = dataset.video_dataset('F:/DATA/', True, mode='Train',stack=False)
        val_dataset = dataset.video_dataset('F:/DATA/', True, mode='Val',stack=False)
    if FLAGS.restore:
        if FLAGS.ckpt_path is not None:
            model.load_state_dict(torch.load(FLAGS.ckpt_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    trainloader = data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
    acc_func=class_acc()
    loss_func=nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr)  # optimize all cnn parameters
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=1e-6)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=FLAGS.warm_epochs, after_scheduler=scheduler_cosine)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title('LOSS',fontsize=10,color='b')
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title('ACC',fontsize=10,color='b')
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title('LR',fontsize=10,color='b')

    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]
    n_epoch=[]
    n_lr=[]

    for epoch in range(FLAGS.epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        n_lr.append(cur_lr)
        print('Epoch %d of %d Start, Current LR is %f'%(epoch+1,FLAGS.epochs,cur_lr))
        n_epoch.append(epoch)
        for phase in ['Train','Val']:
            cnt = 0
            if phase=='Train':
                model.train()
                t_acc=0
                t_loss=0
                for idx,datas in enumerate(tqdm(trainloader)):
                    cnt+=1
                    imgs,labels=datas
                    imgs=imgs.to(device)
                    labels=labels.to(device)
                    optimizer.zero_grad()
                    output=model(imgs)
                    loss = loss_func(output, labels)
                    t_acc+=acc_func(output, labels).detach().cpu().numpy()
                    t_loss += loss.detach().cpu().numpy()
                    loss.backward()
                    print(model.cn12.weight.grad)
                    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 15.)
                    optimizer.step()
                t_acc=t_acc/cnt
                t_loss=t_loss/cnt
                print('Train acc is %f loss is %f'%(t_acc,t_loss))
                train_loss.append(t_loss)
                train_acc.append(t_acc)
            if phase=='Val':

                model.eval()
                with torch.no_grad():
                    v_loss = 0
                    v_acc = 0
                    for idx,datas in enumerate(tqdm(valloader)):
                        cnt += 1
                        imgs,labels=datas
                        imgs=imgs.to(device)
                        labels=labels.to(device)
                        output = model(imgs)
                        loss = loss_func(output, labels)
                        v_acc += acc_func(output, labels).detach().cpu().numpy()
                        v_loss += loss.detach().cpu().numpy()
                v_acc=v_acc/cnt
                v_loss=v_loss/cnt
                print('Val acc is %f loss is %f'%(v_acc,v_loss))
                val_loss.append(v_loss)
                val_acc.append(v_acc)
        ax1.plot(n_epoch, train_loss, '-b', label='Train')
        ax1.plot(n_epoch, val_loss, '-r', label='Val')
        ax2.plot(n_epoch, train_acc, '-b', label='Train')
        ax2.plot(n_epoch, val_acc, '-r', label='Val')
        ax3.plot(n_epoch, n_lr, '-b', label='LR')
        plt.pause(0.1)
        scheduler_warmup.step()
        torch.save(model.state_dict(), str(FLAGS.save_dir)+str(FLAGS.model)+'_'+str(epoch)+'.pt')

if __name__=='__main__':
    app.run(main)
