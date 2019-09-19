import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        #160
        self.cn11=nn.Conv2d(2,32,3,1,1)
        self.bn11=nn.BatchNorm2d(32,affine=False)
        self.cn12=nn.Conv2d(32,32,3,1,1)
        self.bn12=nn.BatchNorm2d(32,affine=False)
        self.cn13=nn.Conv2d(32,32,3,2,1)
        self.bn13=nn.BatchNorm2d(32,affine=False)
        #80
        self.cn21=nn.Conv2d(32,64,3,padding=1)
        self.bn21=nn.BatchNorm2d(64,affine=False)
        self.cn22=nn.Conv2d(64,64,3,2,1)
        self.bn22=nn.BatchNorm2d(64,affine=False)
        #40
        self.cn31=nn.Conv2d(64,64,3,padding=1)
        self.bn31=nn.BatchNorm2d(64,affine=False)
        self.cn32=nn.Conv2d(64,64,3,2,1)
        self.bn32=nn.BatchNorm2d(64,affine=False)
        #20
        self.cn41=nn.Conv2d(64,64,3,padding=1)
        self.bn41=nn.BatchNorm2d(64,affine=False)
        self.cn42=nn.Conv2d(64,128,3,2,1)
        self.bn42=nn.BatchNorm2d(128,affine=False)
        #10
        self.cn5=nn.Conv2d(128,128,3)
        self.bn5=nn.BatchNorm2d(128,affine=False)
        #8
        self.cn6=nn.Conv2d(128,256,3)
        self.bn6=nn.BatchNorm2d(256,affine=False)
        self.cn7=nn.Conv2d(256,512,3)
        self.bn7=nn.BatchNorm2d(512,affine=False)
        self.pool=nn.AvgPool2d(4,1)
        #4
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bias=True,
            bidirectional=True

        )
        self.drop=nn.Dropout(0.5)
        self.out=nn.Linear(256,2)

    def forward(self, x):
        x=x.view(-1,2,x.shape[2],x.shape[3])
        nn=self.cn11(x)
        nn=self.bn11(nn)
        nn=torch.relu(nn)
        nn=self.cn12(nn)
        nn=self.bn12(nn)
        nn=torch.relu(nn)
        nn=self.cn13(nn)
        nn=self.bn13(nn)
        nn=torch.relu(nn)

        nn=self.cn21(nn)
        nn=self.bn21(nn)
        nn=torch.relu(nn)
        nn=self.cn22(nn)
        nn=self.bn22(nn)
        nn=torch.relu(nn)

        nn=self.cn31(nn)
        nn=self.bn31(nn)
        nn=torch.relu(nn)
        nn=self.cn32(nn)
        nn=self.bn32(nn)
        nn=torch.relu(nn)

        nn=self.cn41(nn)
        nn=self.bn41(nn)
        nn=torch.relu(nn)
        nn=self.cn42(nn)
        nn=self.bn42(nn)
        nn=torch.relu(nn)
        nn=self.cn5(nn)
        nn=self.bn5(nn)
        nn=torch.relu(nn)
        nn=self.cn6(nn)
        nn=self.bn6(nn)
        nn=torch.relu(nn)
        nn=self.cn7(nn)
        nn=self.bn7(nn)
        nn=torch.relu(nn)
        nn=self.pool(nn)
        nn=nn.reshape(-1,config.step-1,512)
        r_out, h_n = self.rnn(nn)
        nn=r_out[:,-1, :]
        nn=self.drop(nn)
        nn=self.out(nn)
        nn=F.softmax(nn,1)
        return nn


class C2D(nn.Module):
    def __init__(self):
        super().__init__()
        #160
        self.cn11=nn.Conv2d(config.step,32,3,1,1)
        self.bn11=nn.BatchNorm2d(32,affine=False)
        self.cn12=nn.Conv2d(32,32,3,1,1)
        self.bn12=nn.BatchNorm2d(32,affine=False)
        self.cn13=nn.Conv2d(32,32,3,2,1)
        self.bn13=nn.BatchNorm2d(32,affine=False)
        #80
        self.cn21=nn.Conv2d(32,64,3,padding=1)
        self.bn21=nn.BatchNorm2d(64,affine=False)
        self.cn22=nn.Conv2d(64,64,3,2,1)
        self.bn22=nn.BatchNorm2d(64,affine=False)
        #40
        self.cn31=nn.Conv2d(64,64,3,padding=1)
        self.bn31=nn.BatchNorm2d(64,affine=False)
        self.cn32=nn.Conv2d(64,64,3,2,1)
        self.bn32=nn.BatchNorm2d(64,affine=False)
        #20
        self.cn41=nn.Conv2d(64,64,3,padding=1)
        self.bn41=nn.BatchNorm2d(64,affine=False)
        self.cn42=nn.Conv2d(64,128,3,2,1)
        self.bn42=nn.BatchNorm2d(128,affine=False)
        #10
        self.cn5=nn.Conv2d(128,128,3)
        self.bn5=nn.BatchNorm2d(128,affine=False)
        #8
        self.cn6=nn.Conv2d(128,256,3)
        self.bn6=nn.BatchNorm2d(256,affine=False)
        self.cn7=nn.Conv2d(256,512,3)
        self.bn7=nn.BatchNorm2d(512,affine=False)
        self.pool=nn.AvgPool2d(4,1)
        self.drop=nn.Dropout(0.5)
        self.out=nn.Linear(512,2)

    def forward(self, x):

        nn=self.cn11(x)
        nn=self.bn11(nn)
        nn=torch.relu(nn)
        nn=self.cn12(nn)
        nn=self.bn12(nn)
        nn=torch.relu(nn)
        nn=self.cn13(nn)
        nn=self.bn13(nn)
        nn=torch.relu(nn)

        nn=self.cn21(nn)
        nn=self.bn21(nn)
        nn=torch.relu(nn)
        nn=self.cn22(nn)
        nn=self.bn22(nn)
        nn=torch.relu(nn)

        nn=self.cn31(nn)
        nn=self.bn31(nn)
        nn=torch.relu(nn)
        nn=self.cn32(nn)
        nn=self.bn32(nn)
        nn=torch.relu(nn)

        nn=self.cn41(nn)
        nn=self.bn41(nn)
        nn=torch.relu(nn)
        nn=self.cn42(nn)
        nn=self.bn42(nn)
        nn=torch.relu(nn)
        nn=self.cn5(nn)
        nn=self.bn5(nn)
        nn=torch.relu(nn)
        nn=self.cn6(nn)
        nn=self.bn6(nn)
        nn=torch.relu(nn)
        nn=self.cn7(nn)
        nn=self.bn7(nn)
        nn=torch.relu(nn)
        nn=self.pool(nn)
        nn=nn.reshape(-1,512)
        nn=self.drop(nn)
        nn=self.out(nn)
        nn=F.softmax(nn,1)
        return nn


class C3D(nn.Module):
    def __init__(self):
        super().__init__()
        #160
        self.cn11=nn.Conv3d(1,32,(3,3,3),1,1)
        self.bn11=nn.BatchNorm3d(32,affine=False)
        self.cn12=nn.Conv3d(32,32,(3,3,3),1,1)
        self.bn12=nn.BatchNorm3d(32,affine=False)
        self.cn13=nn.Conv3d(32,32,(3,3,3),(1,2,2),1)
        self.bn13=nn.BatchNorm3d(32,affine=False)
        #80
        self.cn21=nn.Conv3d(32,64,(2,3,3),1,(0,1,1))
        self.bn21=nn.BatchNorm3d(64,affine=False)
        self.cn22=nn.Conv3d(64,64,(2,3,3),(1,2,2),(0,1,1))
        self.bn22=nn.BatchNorm3d(64,affine=False)
        #40
        self.cn31=nn.Conv3d(64,64,3,padding=1)
        self.bn31=nn.BatchNorm3d(64,affine=False)
        self.cn32=nn.Conv3d(64,64,(2,3,3),2,(0,1,1))
        self.bn32=nn.BatchNorm3d(64,affine=False)
        # (config.step-2)/2
        #20
        self.cn41=nn.Conv3d(64,64,(3,3,3),1,(1,1,1))
        self.bn41=nn.BatchNorm3d(64,affine=False)
        self.cn42=nn.Conv3d(64,128,((config.step-2)//2,3,3),(1,2,2),(0,1,1))
        self.bn42=nn.BatchNorm3d(128,affine=False)
        #10
        self.cn5=nn.Conv2d(128,128,3)
        self.bn5=nn.BatchNorm2d(128,affine=False)
        #8
        self.cn6=nn.Conv2d(128,256,3)
        self.bn6=nn.BatchNorm2d(256,affine=False)
        self.cn7=nn.Conv2d(256,512,3)
        self.bn7=nn.BatchNorm2d(512,affine=False)
        self.pool=nn.AvgPool2d(4,1)
        self.drop=nn.Dropout(0.5)
        self.out=nn.Linear(512,2)

    def forward(self, x):
        x = x.view(-1, 1,config.step, x.shape[2], x.shape[3])
        nn=self.cn11(x)
        nn=self.bn11(nn)
        nn=torch.relu(nn)
        nn=self.cn12(nn)
        nn=self.bn12(nn)
        nn=torch.relu(nn)
        nn=self.cn13(nn)
        nn=self.bn13(nn)
        nn=torch.relu(nn)
        nn=self.cn21(nn)
        nn=self.bn21(nn)
        nn=torch.relu(nn)
        nn=self.cn22(nn)
        nn=self.bn22(nn)
        nn=torch.relu(nn)
        nn=self.cn31(nn)
        nn=self.bn31(nn)
        nn=torch.relu(nn)
        nn=self.cn32(nn)
        nn=self.bn32(nn)
        nn=torch.relu(nn)
        nn=self.cn41(nn)
        nn=self.bn41(nn)
        nn=torch.relu(nn)
        nn=self.cn42(nn)
        nn=self.bn42(nn)
        nn=torch.relu(nn)
        nn = nn.view(nn.shape[0], -1, nn.shape[3], nn.shape[4])
        nn=self.cn5(nn)
        nn=self.bn5(nn)
        nn=torch.relu(nn)
        nn=self.cn6(nn)
        nn=self.bn6(nn)
        nn=torch.relu(nn)
        nn=self.cn7(nn)
        nn=self.bn7(nn)
        nn=torch.relu(nn)
        nn=self.pool(nn)
        nn=nn.reshape(-1,512)
        nn=self.drop(nn)
        nn=self.out(nn)
        nn=F.softmax(nn,1)
        return nn

