import torch
import torch.nn as nn

class class_acc(nn.Module):
    def __init__(self):
        super(class_acc, self).__init__()
    def forward(self, predict,target):
        temp=torch.argmax(predict,1)
        temp=temp==target
        temp=torch.sum(temp.float())/temp.shape[0]
        return temp