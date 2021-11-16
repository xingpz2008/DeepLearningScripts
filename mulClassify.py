#This script is used to realize the multi-classification problem, softmax classifier
import torch
import numpy as np

from torchvision import transforms
#Used for img processing
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn.functional as func
#In the multiclassification problem, we try to use softmax to get prob of every res,
#Then we calculate the loss function as P(label)=1, otherwise=0
#Then bp update the weight, so we do not care what the val before softmax is.

#As a result using softmax, the loss function will be changed to
#Loss=-ylogy', where y is label and y' is y_pred, used as torch.nn.NLLLoss
#This function determines the diff between label distribution [0,...,1,...,0] and pred distribution [prob1,prob2,...,probn]
#CrossEntropyLoss includes softmax layer
#CrossEntropyLoss<==>LogSoftmax+NLLLoss

#Now let's start coding

#1.Data Prepration
batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
#This sentence convert PIL Image to Tensor, make pixel val ranging from 0 to 1
#ToTensor(): For image files read by Python API, we will get imageData with (W)idth*(H)eight*(C)hannel, in PyTorch we need C*W*H to process
#   i.e. Z_28*28,[0,255]->R_1*28*28[0,1]
#transform.Compose([func1(),func2(),...]) build a pipeline(or called procedure) which consist of func1,2,...
#transform.Normalize(mean,std) convert all vals into a distribution @ [0,1], which mean is 0.1307 and std of 0.3081
#This 2 vals are calculated from all MNIST samples by calculating the mean and std of all data pixels.

#Let's download and load
train_data=datasets.MNIST(root='./datasets/mnist/',train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='./datasets/mnist/',train=False,download=True,transform=transform) 

train_loader=DataLoader(train_data,shuffle=True,batch_size=batch_size)
test_loader=DataLoader(test_data,shuffle=False,batch_size=batch_size)


#2.Model design
#This script we still use full-connected layer
#The FC requires that input should be a vector, cannot be 1*28*28 matrix
#So we connect every row together to get a 1*784 vector
class multiClsfy(torch.nn.Module):
    def __init__(self):
        super(multiClsfy,self).__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)

    def forward(self,x):
        x=x.view(-1,784)
        #torch.view(outputX,outputY) turn a tensor which size is (A*B) to (outputX,outputY)
        #where outputX*outputY=A*B, if you set one of args to -1, this very -1 will be calculated automatically according to the other arg.
        x=func.relu(self.l1(x))
        x=func.relu(self.l2(x))
        x=func.relu(self.l3(x))
        x=func.relu(self.l4(x))
        x=self.l5(x)
        #Note that the very layer before softmax do not need to be activated.
        return x

model=multiClsfy()

#Because softmax is build in CrossEntropyLoss, we can use it directly
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#We use momentum here to better the training performance

#This time, we build the training process in to a function
def train(epoch):
    total_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        #We can print loss every 300 batches
        #There are lots of iterations in the training, exactly=numData/64
        if batch_idx%300==299:
            print('[%d, %5d] loss= %.3f' % (epoch+1,batch_idx+1,total_loss/300))
            total_loss=0.0

def test():
    correct=0
    total=0
    #Use key word 'with' to avoid grad calculation
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            #output is a 1*10 tensor with 10 probs, we have to check the biggest label
            #torch.max returns maxVal, ValLabel
            _,predicted=torch.max(output.data, dim=1)
            #This ignored the maxVal
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy %d %%' % (100*correct/total))


if __name__=='__main__':
    for epoch in range(100):
        train(epoch)
        if epoch%10==9:
            test()
            #We dont have to test every epoch


