#This script is created for advanced CNN
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.utils.data

#Usually, a complex network is very hard to code line by line
#Instead, we have to build it into a function or a class
#Empirically, we use an 1*1 keneral core to change the channelNum

#Concatenate action is provided as
#torch.cat(outputs,dim=1),where outputs is a list of outputs
#dim=1 indicates that the concatenation follows the channel direction,
#i.e. channelNum remains the same, while length=sumup

#In this script, we focus on famous residual net or ResNet!
class Block(torch.nn.Module):
    def __init__(self,channels):
        #Here we define channels as an input arg
        super(ResNet,self).__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        #Here we use padding to ensure the dim of matrix
